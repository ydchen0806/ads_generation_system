# Advertising Video Generation System

A modular system for generating marketing videos from product images using AI agents.

## Overview

This system uses multiple AI agents to create a complete marketing video from a single product image:

1. **Director Agent**: Analyzes the product image and creates a narrative with keyframe descriptions
2. **Image Agent**: Generates keyframe images based on the director's descriptions
3. **Transition Agent**: Creates smooth transitions between keyframes
4. **Video Agent**: Assembles the final marketing video with keyframes and transitions

## Requirements

- Python 3.8+
- FFmpeg (for video assembly)
- Local deployments of:
  - Qwen model (for director and transition agents)
  - OmniControl model (for image generation)
  - WAN model (for video transitions)

## Installation

```bash
git clone https://github.com/yourusername/ad-generation-system.git
cd ad-generation-system
pip install -r requirements.txt
```
## Usage
1. Place your product image in the `input_images` directory.
2. Run the `main.py` script to start the video generation process.

```bash
python main.py --image_path /path/to/product.jpg --keyframes 5 --output_dir ./output
```
Parameters:
- `--image_path`: Path to the input product image.
- `--keyframes`: Number of keyframes to generate (default: 5).
- `--output_dir`: Directory to save the generated video (default: `./output`).
3. The generated video will be saved in the `output` directory.
4. You can customize the video generation process by modifying the parameters in the `config.py` file.
5. For more advanced usage, you can directly interact with the individual agents by importing them in your scripts.
## Agents
### Director Agent
The Director Agent analyzes the product image and generates a narrative with keyframe descriptions. It uses the Qwen model to understand the image and create a script for the video.
### Image Agent
The Image Agent generates keyframe images based on the descriptions provided by the Director Agent. It uses the OmniControl model to create high-quality images that match the narrative.
### Transition Agent
The Transition Agent creates smooth transitions between keyframes. It uses the WAN model to generate transitions that enhance the flow of the video.
### Video Agent
The Video Agent assembles the final marketing video using FFmpeg. It combines the keyframe images and transitions into a cohesive video that showcases the product.
## Contributing
We welcome contributions to this project! If you have suggestions, bug fixes, or new features, please open an issue or submit a pull request.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Acknowledgments
- Thanks to the developers of the Qwen, OmniControl, and WAN models for their amazing work in AI image generation and video processing.
- Special thanks to the open-source community for their contributions and support.
## Contact
For questions or feedback, please contact us at [cyd0806@mail.ustc.edu.cn]
or open an issue on GitHub.



