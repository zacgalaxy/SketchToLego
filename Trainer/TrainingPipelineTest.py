import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import math
import numpy as np
import random
import gc
from PIL import Image, ImageStat
from itertools import repeat
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import kornia.losses
from torch.utils.checkpoint import checkpoint
import torchvision.models as models


scaler = GradScaler()

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:16].eval()  # Use early layers
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.vgg = vgg

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return torch.nn.functional.mse_loss(pred_features, target_features)


def hook_unet(unet):
    """
    Registers forward hooks on selected layers of the UNet model and returns a list of those layers.
    
    The hook saves the module's output (converted to float) into an attribute called 'output'.
    This output can later be used to extract intermediate features for guidance.

    Args:
        unet: The UNet model instance.

    Returns:
        A list of the hooked modules.
    """
    # Specify which indices (of down, mid, and up blocks) you want to hook.
    blocks_idx = [0, 1, 2]
    feature_blocks = []

    def hook(module, input, output):
        # If the output is a tuple, take the first element.
        if isinstance(output, tuple):
            output = output[0]
        # Convert the output to float and attach it as an attribute.
        setattr(module, "output", output.float())

    # Register hooks on the UNet's down blocks.
    for idx, block in enumerate(unet.down_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block)

    # Register hooks on the mid block's attention and resnet modules.
    for block in unet.mid_block.attentions + unet.mid_block.resnets:
        block.register_forward_hook(hook)
        feature_blocks.append(block)

    # Register hooks on the UNet's up blocks.
    for idx, block in enumerate(unet.up_blocks):
        if idx in blocks_idx:
            block.register_forward_hook(hook)
            feature_blocks.append(block)

    return feature_blocks

def checkpointed_unet_forward(model, *inputs):
    # This function calls your model forward pass (or a subset of it)
    return model(*inputs)

# ----------------------------
# Trainer Class
class SketchGuidedText2ImageTrainer():
    def __init__(self, stable_diffusion_pipeline, unet, vae, text_encoder, lep_unet, scheduler, tokenizer, sketch_simplifier, device):
        self.stable_diffusion_pipeline = stable_diffusion_pipeline
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = device
        self.sketch_simplifier = sketch_simplifier
        self.lep_unet = lep_unet
        self.true_random_state = random.Random()

        # Set models to appropriate modes:
        self.vae.eval()            # VAE is frozen.
        self.text_encoder.eval()   # Text encoder is frozen.
        self.lep_unet.eval()       # ULPE remains frozen.
        self.unet.train()          # Only UNet is updated.
        self.unet.requires_grad_(True)
        self.lep_unet.requires_grad_(False)
        self.vgg_loss = VGGPerceptualLoss().to(self.device)

        # Hook UNet layers to extract intermediate features.
        self.feature_blocks = hook_unet(self.unet)
    def train_step(self, images, sketches, text_prompts, optimizer, noise_scheduler, 
                  num_inference_timesteps, classifier_guidance_strength=8, 
                  sketch_guidance_strength=0.5, guidance_steps_perc=0.5, seed=None):
        " the guideance steps perc was 0.5"
        " the sketch_guidance_strength step was  perc to 1.8"
        
        """
        This training step follows the original inference guidance:
          1. Encode ground-truth images into latents via the frozen VAE.
          2. Compute text embeddings.
          3. Add noise to the image latents.
          4. Iteratively denoise the latents using the UNet.
            At each timestep, intermediate UNet features (collected via hooks) are used
            to compute a guidance edge map via the LEP network.
            For early timesteps, the gradient of the similarity between the LEP output and 
            the target edge map is used to adjust the latents.
          5. The final latent is compared (via MSE loss) with the ground-truth latents.
        """
        if seed:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        gpu=0
        # 1. Encode ground-truth images into latents.
        with torch.no_grad():
            gt_latents = self.encode_to_latents(images)
        
        # 2. Compute text embeddings. can chnage the text encode to use np "_"
        final_text_embeddings = self.stable_diffusion_pipeline._encode_prompt(
            prompt=text_prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        
        # 3. Add noise to the image latents.
        batch_size = images.shape[0]
        noise = torch.randn(batch_size,
                            self.unet.config.in_channels,
                            self.unet.config.sample_size,
                            self.unet.config.sample_size).to(self.device)
      
        # 3.1 Apply diffuion fomrulea to combine image latents and noise
        
        # Random timestep for each sample, either use random nosie or the scheduler noise
       
        randomnoise = self.true_random_state.random() < 0.2
        print(randomnoise)
        if randomnoise:
        # Directly use random noise as latents (simulate inference conditions)
            print("RANDOam noise _________________________-----------------")
            latents = noise.half() * self.scheduler.init_noise_sigma

        else:
            # Your original noise addition logic
            t = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device)
            latents = self.scheduler.add_noise(gt_latents, noise.half(), t)
            

        # Save a clone for guidance computations.
        noise_clone = latents.detach().clone()
        
        # Set the scheduler timesteps (e.g., 12).
        self.scheduler.set_timesteps(num_inference_timesteps)
        
        #choose a random sketch for genralisation
        selected_sketches = [self.true_random_state.choice(sk_list) for sk_list in sketches]

        # Prepare edge maps from sketches (assumed to be working as before).
        encoded_edge_maps = self.prepare_edge_maps(
            prompt=text_prompts,
            num_images_per_prompt=1,
            edge_maps=selected_sketches
        )
        #print("Encoded edge maps norm:", torch.linalg.norm(encoded_edge_maps.float()).item())
        
        # 4. Iterative denoising loop.
        for i, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising")):
            gpu+=torch.cuda.memory_allocated(device=None)
            # Determine gradient state based on timestep.
            if i > int(guidance_steps_perc * num_inference_timesteps):
                current_gradient_state = torch.no_grad()
                gradient = False
            else:
                current_gradient_state = torch.enable_grad()
                gradient = True
            
            # Duplicate latents for classifier-free guidance.
            unet_input = self.scheduler.scale_model_input(torch.cat([latents]*2), timestep).to(self.device)
            unet_input = unet_input.requires_grad_(True)
            
            # Forward pass through UNet.
            with current_gradient_state:
                output = self.unet(unet_input, timestep, encoder_hidden_states=final_text_embeddings).sample
                u, t = output.chunk(2)

            pred = u + classifier_guidance_strength * (t - u)
            latents_old = unet_input.chunk(2)[1]
            latents = self.scheduler.step(pred, timestep, latents).prev_sample


            # Guidance branch.
            with current_gradient_state:
                intermediate_result = []
                fixed_size = latents.shape[2]
                for block in self.feature_blocks:
                    outputs = block.output  # Use the hook outputs as stored
                    resized = F.interpolate(outputs, size=fixed_size, mode="bilinear")
                    intermediate_result.append(resized)
                    del block.output
                    # Do not clear the outputs so that the UNet's internal structure is not affected.
                intermediate_result = torch.cat(intermediate_result, dim=1).to(self.device)
                
                # Obtain noise level.
                noise_level = self.get_noise_level(noise_clone, timestep)
                # In the original pipeline, they concatenate two copies of noise_level.
                # (Assuming noise_level already has the expected shape; if not, adjust accordingly.)
                lep_input = torch.cat([noise_level, noise_level]).to(self.device)
                
                # Run the LEP network.
                result = self.lep_unet(intermediate_result, lep_input)
                _, result = result.chunk(2)
                
                if gradient:
                    similarity = torch.linalg.norm(result - encoded_edge_maps)**2
                    # Compute gradients from the similarity.
                    _, grad = torch.autograd.grad(similarity, unet_input, retain_graph=True)[0].chunk(2)
                    
                    """Remeberer This is for adpative scaling the eplislon here from 1e-4
                    grad_norm = torch.linalg.norm(grad) + 1e-6  # Avoid division by zero
                    scaling_factor = torch.sigmoid(grad_norm / 10)  # Adaptive scaling
                    alpha = (torch.linalg.norm(latents_old - latents) / grad_norm) * sketch_guidance_strength * scaling_factor
                    """
                    alpha = (torch.linalg.norm(latents_old - latents)/torch.linalg.norm(grad))*sketch_guidance_strength
                    latents = latents - alpha* grad
                    if torch.isnan(latents).any() or torch.isnan(grad).any():
                     print("NaN detected; skipping update or breaking.")
                     #print("unet_input norm:", torch.linalg.norm(unet_input).item())
                     #print("latents_old norm:", torch.linalg.norm(latents_old).item())
                     #print("latents norm:", torch.linalg.norm(latents).item())
                     #print("Encoded edge maps norm:", torch.linalg.norm(encoded_edge_maps.float()).item())
                    
                    #print(f"Step {i} | timestep: {timestep} | alpha: {alpha.item():.4f}")
            
            gc.collect()
            torch.cuda.empty_cache()

        # 5. Compute final MSE loss.
        latentstoimage = self.latents_to_image(latents)
        edges=self.latents_to_image(encoded_edge_maps)
        """
        for edge, image in zip(edges, latentstoimage):
            fig, axs = plt.subplots(1, 2, figsize = (10, 5))
            axs[0].imshow(edge)
            axs[1].imshow(image)
            axs[0].axis("off")
            axs[1].axis("off")
            axs[0].set_title("edge map")
            axs[1].set_title(f"{text_prompts}")
        plt.show()
        """

        
        guidance_loss = torch.linalg.norm(result - encoded_edge_maps) ** 2  
        #mask = self.create_mask(images)  # Convert images into a mask
        #loss = self.masked_mse_loss(latents, gt_latents, mask)

        loss= F.mse_loss(latents, gt_latents)
        ssim_loss = self.compute_ssim_loss(
            self.image_to_tensor(latentstoimage[0]),  # Convert PIL to Tensor
            images
        )
        edge_loss = self.edge_preservation_loss(
            self.image_to_tensor(latentstoimage[0]),  # Convert PIL to Tensor
            images
        )
        whiteloss= self.white_space_penalty(self.image_to_tensor(latentstoimage[0]))
       
        pred_vgg = self.preprocess_for_vgg(self.image_to_tensor(latentstoimage[0]))
        gt_vgg = self.preprocess_for_vgg(images)
        vgg= self.vgg_loss(pred_vgg, gt_vgg)
        return loss, guidance_loss, (gpu/timestep), ssim_loss, edge_loss, whiteloss,vgg
                                    
                                    
    
    @torch.no_grad()
    def encode_to_latents(self, images):
        """
        Encodes image tensors (with pixel values in [0,1]) into latent representations using the frozen VAE.
        """
        images = images.to(self.vae.dtype)
        images = images * 2 - 1  # Normalize to [-1,1]
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        return latents

    def image_to_tensor(self, img):
      """
      Convert PIL Image to Torch Tensor (in the range [0,1])
      """
      transform = transforms.Compose([
          transforms.ToTensor()  # Converts PIL Image to Tensor (C, H, W)
      ])
      return transform(img).unsqueeze(0).to(self.device)  # Add batch dimension

    @torch.no_grad()
    def text_to_embeddings(self, text):
        """
        Converts text prompts to embeddings.
        """
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        embeddings = self.text_encoder(tokenized.input_ids)[0]
        return embeddings.to(self.unet.dtype)

    def compute_alphas_cumprod(self, scheduler):
      timesteps = scheduler.num_train_timesteps
      betas = torch.linspace(scheduler.beta_start, scheduler.beta_end, timesteps)
      alphas = 1 - betas
      alphas_cumprod = torch.cumprod(alphas, dim=0)
      return alphas_cumprod
    
    def get_noise_level(self, noise, timesteps):
        """
        Computes a noise level tensor based on the scheduler's alphas_cumprod.
        """
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise_level = sqrt_one_minus_alpha_prod.to(self.device) * noise
        return noise_level
    
    def create_mask(self, image, threshold=0.95):
        """
        Create a binary mask where white pixels (background) are 0 and everything else is 1.
        Expects `image` to be a tensor of shape [B, C, H, W] with values in [0, 1].
        """
        # Convert RGB to grayscale by averaging across channels (assuming C=3)
        grayscale = image.mean(dim=1, keepdim=True)  
        mask = (grayscale < threshold).float()  
        return mask
    
    def masked_mse_loss(self, pred, target, mask):
        return F.mse_loss(pred * mask, target * mask)

    @torch.no_grad()
    def image_to_latents(self, img, img_type):
        """
        Converts a PIL image to a latent representation using the frozen VAE.
        If img_type is "edge_map", the image may be binarized.
        """

        np_img = np.array(img).astype(np.float16) / 255.0
        if img_type == "edge_map":
            np_img[np_img < 0.5] = 0.
            np_img[np_img >= 0.5] = 1.
        np_img = np_img * 2.0 - 1.0
        # If the image is grayscale (2D), add a channel dimension.
        if np_img.ndim == 2:
            np_img = np.expand_dims(np_img, axis=-1)  # shape becomes (H, W, 1)
        # Now add the batch dimension and transpose to NCHW format.
        np_img = np_img[None].transpose(0, 3, 1, 2)  # shape becomes (1, C, H, W)
        torch_img = torch.from_numpy(np_img)
        generator = torch.Generator(self.device).manual_seed(0)
        latents = self.vae.encode(torch_img.to(self.vae.dtype).to(self.device)).latent_dist.sample(generator=generator)
        latents = latents * 0.18215
        return latents

    
    @torch.no_grad()
    def prepare_edge_maps(self, prompt, num_images_per_prompt, edge_maps):
        batch_size = len(prompt)
        size=512
        #size=256
        if batch_size != len(edge_maps):
            raise ValueError("Wrong number of edge maps")
            
        processed_edge_maps = []
        for edge_map in edge_maps:
            arr = np.array(edge_map)
            # If the image is grayscale (2D), use it as is (after inversion).
            if arr.ndim == 2:
                inverted = 255 - arr
            else:
                inverted = 255 - arr[:,:,:3]
            processed_img = Image.fromarray(inverted).resize((size, size))
            processed_edge_maps.append(processed_img)
            
        encoded_edge_maps = [self.image_to_latents(edge.resize((size,size)), img_type="edge_map") 
                            for edge in processed_edge_maps]
        # Repeat each edge map if necessary.
        encoded_edge_maps_final = [edge for edge in encoded_edge_maps for _ in range(num_images_per_prompt)]
        encoded_edge_maps_tensor = torch.cat(encoded_edge_maps_final, dim=0)
        return encoded_edge_maps_tensor
    
    @torch.no_grad()
    def latents_to_image(self, latents):
        """
        Decodes image latents into PIL images using the frozen VAE decoder.
        """
        latents = (1/0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image/2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(img) for img in images]
        return pil_images
    
    def compute_ssim_loss(self, pred, target):
        return 1 - kornia.losses.ssim_loss(pred, target, window_size=5, reduction="mean")

    def rgb_to_grayscale(self, img):
      """Convert an RGB image tensor to grayscale."""
      return img.mean(dim=1, keepdim=True)  # Averages across channels (C=3 → C=1)
      
    def edge_preservation_loss(self, pred, target):
        """Apply edge preservation loss by first converting to grayscale."""
        
        # ✅ Convert RGB to Grayscale (Retains Shape: [B, C, H, W] but now C=1)
        pred_gray = self.rgb_to_grayscale(pred)
        target_gray = self.rgb_to_grayscale(target)

        # ✅ Define Sobel Filter (Edge Detector)
        sobel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).expand(1, 1, 3, 3)

        # ✅ Compute Edge Maps
        pred_edges = F.conv2d(pred_gray, sobel, padding=1)
        target_edges = F.conv2d(target_gray, sobel, padding=1)

        # ✅ Compute Edge Loss
        return F.mse_loss(pred_edges, target_edges)
    def white_space_penalty(self, image):
      white_threshold = 0.90  # pixels nearly pure white
      penalty = (image > white_threshold).float().mean()
      return penalty
    def preprocess_for_vgg(self, img_tensor):
      # Expects input in range [0,1], shape (B, C, H, W)
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
      return normalize(img_tensor)

    @torch.no_grad()
    def eval(self, images, sketches, text_prompts, optimizer, noise_scheduler, 
                  num_inference_timesteps, classifier_guidance_strength=8, 
                  sketch_guidance_strength=0.5, guidance_steps_perc=0.5, seed=None):
        " the guideance steps perc was 0.5"
        " the sketch_guidance_strength step was  perc to 1.8"
        
        """
        This training step follows the original inference guidance:
          1. Encode ground-truth images into latents via the frozen VAE.
          2. Compute text embeddings.
          3. Add noise to the image latents.
          4. Iteratively denoise the latents using the UNet.
            At each timestep, intermediate UNet features (collected via hooks) are used
            to compute a guidance edge map via the LEP network.
            For early timesteps, the gradient of the similarity between the LEP output and 
            the target edge map is used to adjust the latents.
          5. The final latent is compared (via MSE loss) with the ground-truth latents.
        """
        
        if seed:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        # 1. Encode ground-truth images into latents.
        with torch.no_grad():
            gt_latents = self.encode_to_latents(images)
        
        # 2. Compute text embeddings. can chnage the text encode to use np "_"
        final_text_embeddings = self.stable_diffusion_pipeline._encode_prompt(
            prompt=text_prompts,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        
        # 3. Add noise to the image latents.
        batch_size = images.shape[0]
        noise = torch.randn(batch_size,
                            self.unet.config.in_channels,
                            self.unet.config.sample_size,
                            self.unet.config.sample_size).to(self.device)
      
        latents = noise.half() * self.scheduler.init_noise_sigma




        # Save a clone for guidance computations.
        noise_clone = latents.detach().clone()
        
        # Set the scheduler timesteps (e.g., 12).
        self.scheduler.set_timesteps(num_inference_timesteps)
    

        # Prepare edge maps from sketches (assumed to be working as before).
        encoded_edge_maps = self.prepare_edge_maps(
            prompt=text_prompts,
            num_images_per_prompt=1,
            edge_maps=sketches
        )
        
        # 4. Iterative denoising loop.
        for i, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising")):
            # Determine gradient state based on timestep.
            if i > int(guidance_steps_perc * num_inference_timesteps):
                current_gradient_state = torch.no_grad()
                gradient = False
            else:
                current_gradient_state = torch.enable_grad()
                gradient = True
            
            # Duplicate latents for classifier-free guidance.
            unet_input = self.scheduler.scale_model_input(torch.cat([latents]*2), timestep).to(self.device)
            unet_input = unet_input.requires_grad_(True)

            # Forward pass through UNet.
            with current_gradient_state:
                output = self.unet(unet_input, timestep, encoder_hidden_states=final_text_embeddings).sample
                u, t = output.chunk(2)

            pred = u + classifier_guidance_strength * (t - u)
            latents_old = unet_input.chunk(2)[1]
            latents = self.scheduler.step(pred, timestep, latents).prev_sample


            # Guidance branch.
            with current_gradient_state:
                intermediate_result = []
                fixed_size = latents.shape[2]
                for block in self.feature_blocks:
                    outputs = block.output  # Use the hook outputs as stored
                    resized = F.interpolate(outputs, size=fixed_size, mode="bilinear")
                    intermediate_result.append(resized)
                    del block.output
                    # Do not clear the outputs so that the UNet's internal structure is not affected.
                intermediate_result = torch.cat(intermediate_result, dim=1).to(self.device)
                
                # Obtain noise level.
                noise_level = self.get_noise_level(noise_clone, timestep)
                # In the original pipeline, they concatenate two copies of noise_level.
                # (Assuming noise_level already has the expected shape; if not, adjust accordingly.)
                lep_input = torch.cat([noise_level, noise_level]).to(self.device)
                
                # Run the LEP network.
                result = self.lep_unet(intermediate_result, lep_input)
                _, result = result.chunk(2)
                
                if gradient:
                    similarity = torch.linalg.norm(result - encoded_edge_maps)**2
                    # Compute gradients from the similarity.
                    _, grad = torch.autograd.grad(similarity, unet_input)[0].chunk(2)
                    
                    """Remeberer This is for adpative scaling the eplislon here from 1e-4
                    grad_norm = torch.linalg.norm(grad) + 1e-6  # Avoid division by zero
                    scaling_factor = torch.sigmoid(grad_norm / 10)  # Adaptive scaling
                    alpha = (torch.linalg.norm(latents_old - latents) / grad_norm) * sketch_guidance_strength * scaling_factor
                    """
                    alpha = (torch.linalg.norm(latents_old - latents)/torch.linalg.norm(grad))*sketch_guidance_strength
                    latents = latents - alpha* grad
                    if torch.isnan(latents).any() or torch.isnan(grad).any():
                     print("NaN detected; skipping update or breaking.")
                     #print("unet_input norm:", torch.linalg.norm(unet_input).item())
                     #print("latents_old norm:", torch.linalg.norm(latents_old).item())
                     #print("latents norm:", torch.linalg.norm(latents).item())
                     #print("Encoded edge maps norm:", torch.linalg.norm(encoded_edge_maps.float()).item())
                    
                    #print(f"Step {i} | timestep: {timestep} | alpha: {alpha.item():.4f}")
            
            gc.collect()
            torch.cuda.empty_cache()

        # 5. Compute final MSE loss.
        latentstoimage = self.latents_to_image(latents)
        edges=self.latents_to_image(encoded_edge_maps)
        for edge, image in zip(edges, latentstoimage):
            fig, axs = plt.subplots(1, 2, figsize = (10, 5))
            axs[0].imshow(edge)
            axs[1].imshow(image)
            axs[0].axis("off")
            axs[1].axis("off")
            axs[0].set_title("edge map")
            axs[1].set_title(f"{text_prompts}")
        plt.show()

        
        guidance_loss = torch.linalg.norm(result - encoded_edge_maps) ** 2  
        #mask = self.create_mask(images)  # Convert images into a mask
        #loss = self.masked_mse_loss(latents, gt_latents, mask)
        
        loss= F.mse_loss(latents, gt_latents)
        ssim_loss = self.compute_ssim_loss(
            self.image_to_tensor(latentstoimage[0]),  # Convert PIL to Tensor
            images
        )
        edge_loss = self.edge_preservation_loss(
            self.image_to_tensor(latentstoimage[0]),  # Convert PIL to Tensor
            images
        )
        whiteloss= self.white_space_penalty(self.image_to_tensor(latentstoimage[0]))
       
        pred_vgg = self.preprocess_for_vgg(self.image_to_tensor(latentstoimage[0]))
        gt_vgg = self.preprocess_for_vgg(images)
        vgg= self.vgg_loss(pred_vgg, gt_vgg)
        return loss, guidance_loss, ssim_loss, edge_loss, whiteloss,vgg, latentstoimage[0], edges[0] 
