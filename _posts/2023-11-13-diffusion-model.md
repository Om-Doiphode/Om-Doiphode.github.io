---
layout: post
title: Stable Diffusion Deep Dive
subtitle: 
gh-repo: Om-Doiphode/Diffusion
gh-badge: [star, fork, follow]
tags: [Deep Learning, Generative models]
comments: true
---

Stable Diffusion is a powerful text-to-image model. There are various websites and tools to make using it as easy as possible. It is also integrated into the Huggingface diffusers library.

In the notebook `stable_diffusion.ipynb`, we will begin by recreating the functionality above as a scary code, and then one by one we'll inspect the different components and figure out what they do.

# Components in a Stable Diffusion model

##  Autoencoder (AE)

The autoencoder can 'encode' an input image into some sort of latent representation and decode this back into an image. This is done to reduce the memory requirement for generating an image from the input image.

```python
def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()
```
- <b>Input</b>: The function takes an image as input in PIL format.

- <b>Data Transformation</b>: The input image is transformed into a PyTorch tensor using `tfms.ToTensor()`.

- <b>Scaling</b>: The tensor is then scaled to the range [-1,1] by multiplying it by 2 and subtracting 1. This scaling is done to normalize pixel values.

- <b>Encoder</b>:  The scaled tensor is passed through an encoder (vae.encode(...)) associated with a variational autoencoder (VAE).

- <b>Sampling from latent space</b>: `latent.latent_dist.sample()` : The code then samples a point from the latent distribution obtained from the VAE's encoder. This step involves drawing a sample from a probability distribution, which represents a point in the latent space where the input image is encoded.

- <b>Scaling the Latent Space Point</b>: The sampled latent point is then scaled by a factor of 0.18215. The reason for this scaling factor depends on the specific implementation details of the VAE and its training process.


```python
def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images
```

- <b>Input (latents)</b>: The function takes a batch of latent representations (`latents`) as input. These latents are presumably obtained from a latent space, possibly as a result of encoding images using a variational autoencoder (VAE) or a similar model.

- <b>Scaling Latents</b>: The input latents are scaled by dividing them by the constant value `0.18215`. This appears to be the inverse of the scaling factor used in the `pil_to_latent` function, suggesting a reciprocal relationship between the scaling factors during encoding and decoding.

- <b>Decoder (`vae.decode(latents).sample`)</b>: The scaled latents are then passed through the decoder of a VAE (`vae.decode(latents)`). The decoder is responsible for generating images from points in the latent space. The .sample call indicates that a sample is drawn from the distribution represented by the decoder. This is common in probabilistic generative models like VAEs.

- <b>Image Scaling and Clamping</b>: The generated image is then rescaled to the range [0, 1] by dividing by 2 and adding 0.5. The .clamp(0, 1) operation ensures that pixel values are within the valid range. This step is often necessary when working with neural networks that generate images, as it ensures the pixel values are suitable for display.

- <b>Conversion to NumPy and Integer Range</b>: The image tensor is converted to a NumPy array, and pixel values are rescaled to the range [0, 255] by multiplying by 255. The rounding is applied to ensure that pixel values are integers.

- <b>Conversion to PIL Images</b>: Finally, the NumPy array is used to create a list of PIL images. Each image in the list is created from the corresponding NumPy array.


## Scheduler

During training, we add some noise to an image an then have the model try to predict the noise. If we always added a ton of noise, the model might not have much to work with. If we only add a tiny amount, the model won't be able to do much with the random starting points we use for sampling. So during training the amount is varied, according to some distribution.

During sampling, we want to 'denoise' over a number of steps. How many steps and how much noise we should aim for at each step are going to affect the final result.

The scheduler is in charge of handling all of these details. For example: `scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)` sets up a scheduler that matches the one used to train this model.

- <b>beta_start</b> and <b>beta_end</b>: These parameters define the starting and ending values of the learning rate (or another parameter, denoted as beta). The learning rate starts at 0.00085 and ends at 0.012. During training, the learning rate will be adjusted between these two values.

- <b>beta_schedule</b>: This parameter specifies the type of schedule used to anneal the learning rate. In this case, it's set to "scaled_linear". The schedule type determines how the learning rate changes over time. A scaled linear schedule likely means that the learning rate changes linearly but might be scaled or adjusted in a specific way.

- <b>num_train_timesteps</b>: This parameter sets the total number of training timesteps or iterations. The scheduler will adjust the learning rate over these iterations according to the specified schedule.


```python
# Settings (same as before except for the new prompt)
prompt = ["A colorful dancer, nat geo photo"]
height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 50            # Number of denoising steps
guidance_scale = 8                  # Scale for classifier-free guidance
generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
batch_size = 1

# Prep text (same as before)
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Prep Scheduler (setting the number of inference steps)
set_timesteps(scheduler, num_inference_steps)

# Prep latents (noising appropriately for start_step)
start_step = 10
start_sigma = scheduler.sigmas[start_step]
noise = torch.randn_like(encoded)
latents = scheduler.add_noise(encoded, noise, timesteps=torch.tensor([scheduler.timesteps[start_step]]))
latents = latents.to(torch_device).float()

# Loop
for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
    if i >= start_step: # << This is the only modification to the loop we do

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

latents_to_pil(latents)[0]
```

If we increase the `start_steps`, the output image leans more towards the input image rather than the prompt. Because if we increase the `start_step`, we are essentially starting the guidance and diffusion process later in the sequence of timesteps. This means that, at the beginning of the inference process, the latent noise is more influenced by the initial conditions (input image) rather than the guidance from the prompt.

To address this and ensure that the output is more influenced by the prompt, one might consider starting the guidance and diffusion process earlier by setting a lower value for `start_step`. This way, the model has more opportunity to be guided by the information from the prompt from the beginning of the inference process.


## Exploring the Text -> Embedding pipeline

```python
# Our text prompt
prompt = 'A picture of a puppy'
# Turn the text into a sequnce of tokens:
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_input['input_ids'][0] # View the tokens

# output:
tensor([49406,   320,  1674,   539,   320,  6829, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407])
```

The repeated occurrences of `49407` at the end of the sequence likely represent padding tokens. Padding is often added to ensure that all input sequences have the same length, which is a common requirement for efficient batch processing in deep learning.

`49406` and `49407` represent the start and end of the sequence or other special tokens.

To get the actual tokens:
```python
actual_tokens = text_input['input_ids'][0][:text_input['attention_mask'].sum()]
print(actual_tokens)

# Output:
tensor([49406,   320,  1674,   539,   320,  6829, 49407])
```


Output Embeddings:
```python
# Grab the output embeddings
output_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
print('Shape:', output_embeddings.shape)
output_embeddings

# Output:
Shape: torch.Size([1, 77, 768])

tensor([[[-0.3884,  0.0229, -0.0522,  ..., -0.4899, -0.3066,  0.0675],
         [ 0.0290, -1.3258,  0.3085,  ..., -0.5257,  0.9768,  0.6652],
         [ 0.6942,  0.3538,  1.0991,  ..., -1.5716, -1.2643, -0.0121],
         ...,
         [-0.0221, -0.0053, -0.0089,  ..., -0.7303, -1.3830, -0.3011],
         [-0.0062, -0.0246,  0.0065,  ..., -0.7326, -1.3745, -0.2953],
         [-0.0536,  0.0269,  0.0444,  ..., -0.7159, -1.3634, -0.3075]]],
       device='cuda:0', grad_fn=<NativeLayerNormBackward0>)
```


We pass the tokens to the `text_encoder` and we get some numbers which we can feed to the model. These tokens are transformed into a set of input embeddings, which are then fed through the transformer model to get the final output embeddings.

To get the input embeddings, there are actually two steps -
## 1. Token embedding

This is used to convert the token into a vector of numbers.

```python
# Access the embedding layer
token_emb_layer = text_encoder.text_model.embeddings.token_embedding
token_emb_layer # Vocab size 49408, emb_dim 768
```

A token can be embedded as follows:
```python
# Embed a token - in this case the one for 'puppy'
embedding = token_emb_layer(torch.tensor(6829, device=torch_device))
embedding.shape # 768-dim representation

# Output:
torch.Size([768])
```
Here the token has been mapped to a 786 dimensional vector(the token embedding)

Do the same for all of the tokens in the prompt to get all the token embeddings:
```python
token_embeddings = token_emb_layer(text_input.input_ids.to(torch_device))
print(token_embeddings.shape) # batch size 1, 77 tokens, 768 values for each
token_embeddings

# Output:
torch.Size([1, 77, 768])

tensor([[[ 0.0011,  0.0032,  0.0003,  ..., -0.0018,  0.0003,  0.0019],
         [ 0.0013, -0.0011, -0.0126,  ..., -0.0124,  0.0120,  0.0080],
         [ 0.0235, -0.0118,  0.0110,  ...,  0.0049,  0.0078,  0.0160],
         ...,
         [ 0.0012,  0.0077, -0.0011,  ..., -0.0015,  0.0009,  0.0052],
         [ 0.0012,  0.0077, -0.0011,  ..., -0.0015,  0.0009,  0.0052],
         [ 0.0012,  0.0077, -0.0011,  ..., -0.0015,  0.0009,  0.0052]]],
       device='cuda:0', grad_fn=<EmbeddingBackward0>)
```

## 2. Positional embedding

Positional embedding tell the model the position of a token in a sequence.

To get the positional encoding for each position:
```python
position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]
position_embeddings = pos_emb_layer(position_ids)
print(position_embeddings.shape)
position_embeddings

# Output:
torch.Size([1, 77, 768])

tensor([[[ 0.0016,  0.0020,  0.0002,  ..., -0.0013,  0.0008,  0.0015],
         [ 0.0042,  0.0029,  0.0002,  ...,  0.0010,  0.0015, -0.0012],
         [ 0.0018,  0.0007, -0.0012,  ..., -0.0029, -0.0009,  0.0026],
         ...,
         [ 0.0216,  0.0055, -0.0101,  ..., -0.0065, -0.0029,  0.0037],
         [ 0.0188,  0.0073, -0.0077,  ..., -0.0025, -0.0009,  0.0057],
         [ 0.0330,  0.0281,  0.0289,  ...,  0.0160,  0.0102, -0.0310]]],
       device='cuda:0', grad_fn=<EmbeddingBackward0>)
```

Combine token and positional embeddings as follows:
```python
# And combining them we get the final input embeddings
input_embeddings = token_embeddings + position_embeddings
print(input_embeddings.shape)
```

This gives the same result we'd get from `text_encoder.text_model.embeddings`

```python
# The following combines all the above steps (but doesn't let us fiddle with them!)
text_encoder.text_model.embeddings(text_input.input_ids.to(torch_device))
```

Refer the notebook for more details.
