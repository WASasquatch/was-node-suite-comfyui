# WAS's Comprehensive Node Suite [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FWASasquatch%2Fwas-node-suite-comfyui&count_bg=%233D9CC8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

<p align="center">
    <img src="https://user-images.githubusercontent.com/1151589/228982359-4a6215cc-3ca9-4c24-8a7b-d229d7bce277.png">
</p>

### A node suite for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with many new nodes, such as image processing, text processing, and more. 

### [Share Workflows](/workflows/README.md) to the `/workflows/` directory. Preferably embedded PNGs with workflows, but JSON is OK too. [You can use this tool to add a workflow to a PNG file easily](https://colab.research.google.com/drive/1hQMjNUdhMQ3rw1Wcm3_umvmOMeS_K4s8?usp=sharing)


# Current Nodes:
 - CLIPTextEncode (NSP): Parse Noodle Soup Prompts
 - Constant Number
 - Debug to Console (Debug pretty much anything to the console window)
 - Image Analyze
   - Black White Levels
   - RGB Levels
     - Depends on `matplotlib`, will attempt to install on first run
 - Image Blank: Create a blank image in any color
 - Image Blend by Mask: Blend two images by a mask
 - Image Blend: Blend two images by opacity
 - Image Blending Mode: Blend two images by various blending modes
 - Image Bloom Filter: Apply a high-pass based bloom filter
 - Image Canny Filter: Apply a canny filter to a image
 - Image Chromatic Aberration: Apply chromatic aberration lens effect to a image like in sci-fi films, movie theaters, and video games
 - Image Color Palette
   - Generate a color palette based on the input image. 
     - Depends on `scikit-learn`, will attempt to install on first run. 
   - Supports color range of 8-256
   - Utilizes font in `./res/` unless unavailable, then it will utilize internal better then nothing font. 
 - Image Edge Detection Filter: Detect edges in a image
 - Image Film Grain: Apply film grain to a image
 - Image Filter Adjustments: Apply various image adjustments to a image
 - Image Flip: Flip a image horizontal, or vertical
 - Image Gradient Map: Apply a gradient map to a image
 - Image Generate Gradient: Generate a gradient map with desired stops and colors
 - Image High Pass Filter: Apply a high frequency pass to the image returning the details
 - Image Levels Adjustment: Adjust the levels of a image
 - Image Load: Load a *image* from any path on the system, or a url starting with `http`
 - Image Median Filter: Apply a median filter to a image, such as to smooth out details in surfaces
 - Image Mix RGB Channels: Mix together RGB channels into a single iamge
 - Image Monitor Effects Filter: Apply various monitor effects to a image
   - Digital Distortion
     - A digital breakup distortion effect
   - Signal Distortion
     - A analog signal distortion effect on vertical bands like a CRT monitor
   - TV Distortion
     - A TV scanline and bleed distortion effect
 - Image Nova Filter: A image that uses a sinus frequency to break apart a image into RGB frequencies
 - Image Perlin Noise Filter
   - Create perlin noise with [pythonperlin](https://pypi.org/project/pythonperlin/) module. Trust me, better then my implementations that took minutes... 
 - Image Remove Background (Alpha): Remove the background from a image by threshold and tolerance. 
 - Image Remove Color: Remove a color from a image and replace it with another
 - Image Resize
 - Image Rotate: Rotate an image
 - Image Save: A save image node with format support and path support. (Bug: Doesn't display image
 - Image Seamless Texture: Create a seamless texture out of a image with optional tiling
 - Image Select Channel: Select a single channel of an RGB image
 - Image Select Color: Return the select image only on a black canvas
 - Image Style Filter: Style a image with Pilgram instragram-like filters
   - Depends on `pilgram` module
 - Image Threshold: Return the desired threshold range of a image
 - Image Transpose
 - Image fDOF Filter: Apply a fake depth of field effect to an image
 - Image to Latent Mask: Convert a image into a latent mask
 - Image Voronoi Noise Filter
   - A custom implementation of the worley voronoi noise diagram
 - Input Switch  (Disable until `*` wildcard fix)
 - KSampler (WAS): A sampler that accepts a seed as a node inpu
 - Load Text File
 - Load Batch Images
   - Increment images in a folder, or fetch a single image out of a batch.
   - Will reset it's place if the path, or pattern is changed.
   - pattern is a glob that allows you to do things like `**/*` to get all files in the directory and subdirectory
     or things like `*.jpg` to select only JPEG images in the directory specified. 
 - Latent Noise Injection: Inject latent noise into a latent image
 - Latent Upscale by Factor: Upscale a latent image by a facto
 - MiDaS Depth Approximation: Produce a depth approximation of a single image input
 - MiDaS Mask Image: Mask a input image using MiDaS with a desired color
 - Number Operation
 - Number to Seed
 - Number to Float
 - Number to Int
 - Number to String
 - Number to Text
 - Random Number
 - Save Text File: Save a text string to a file
 - Seed: Return a seed
 - Tensor Batch to Image: Select a single image out of a latent batch for post processing with filters
 - Text Add Tokens: Add custom tokens to parse in filenames or other text.
 - Text Concatenate: Merge two strings
 - Text Find and Replace: Find and replace a substring in a string
 - Text Multiline: Write a multiline text strin
 - Text Parse Noodle Soup Prompts: Parse NSP in a text input
 - Text Parse Tokens: Parse custom tokens in text.
 - Text Random Line: Select a random line from a text input string
 - Text String: Write a single line text string value
 - Text to Conditioning: Convert a text string to conditioning.
 
 
## Text Tokens
Text tokens can be used in the Save Text File and Save Image nodes. You can also add your own custom tokens with the Text Add Tokens node.

The token name can be anything excluding the `:` character to define your token. It can also be simple Regular Expressions.

### Built-in Tokens
  - [time]
    - The current system microtime
  - [hostname]
    - The hostname of the system executing ComfyUI
  - [user]
    - The user that is executing ComfyUI

## Recommended Installation:
If you're running on Linux, or non-admin account on windows you'll want to ensure `/ComfyUI/custom_nodes`, `was-node-suite-comfyui`, and `WAS_Node_Suite.py` has write permissions.

  - Navigate to your `/ComfyUI/custom_nodes/` folder
  - `git clone https://github.com/WASasquatch/was-node-suite-comfyui/`
  - Start ComfyUI
    - WAS Suite should uninstall legacy nodes automatically for you.
    - Tools will be located in the WAS Suite menu.
    
## Alternate Installation:
If you're running on Linux, or non-admin account on windows you'll want to ensure `/ComfyUI/custom_nodes`, and `WAS_Node_Suite.py` has write permissions.

  - Download `WAS_Node_Suite.py`
  - Move the file to your `/ComfyUI/custom_nodes/` folder
  - Start, or Restart ComfyUI
    - WAS Suite should uninstall legacy nodes automatically for you.
    - Tools will be located in the WAS Suite menu.
	
## Installing on Colab
Create a new cell and add the following code, then run the cell.

  - `git clone https://github.com/WASasquatch/was-node-suite-comfyui /content/ComfyUI/custom_nodes/`
  - Restart Colab Runtime (don't disconnect)
    - WAS Suite should uninstall legacy nodes automatically for you.
    - Tools will be located in the WAS Suite menu.

      
### Dependencies:
WAS Node Suite is designed to download dependencies on it's own as needed, but what it depends on can be installed manually before use to prevent any script issues. The dependencies which are not required by ComfyUI are as follows: 
  - opencv
  - scipy
  - pilgram
  - timm (for MiDaS)
    - MiDaS Models (they will download automatically upon use and be stored in `/ComfyUI/models/midas/checkpoints/`, additional files may be installed by `PyTorch Hub`)

*This version renames some nodes, as well as introduces new fields. Unfortunately ComfyUI doesn't handle these changes well, so you'll have to replace the dreaded "red nodes" manually.*
