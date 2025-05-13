# pip install --force-reinstall flash-attn -i https://pypi.tuna.tsinghua.edu.cn/simple
# cd /h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/flash-attention-main/flash-attention-main/hopper
# python setup.py install
# pip install "xfuser>=0.4.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash_attn==2.6.3 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
python /h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/generate.py --task flf2v-14B \
    --size 1280*720 --ckpt_dir /h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/model/Wan2.1-FLF2V-14B-720P \
    --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png \
    --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
    --save_path /h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/video_result_car
    
python /h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/generate.py --task flf2v-14B --size 1280*720 \
    --ckpt_dir /h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/model/Wan2.1-FLF2V-14B-720P --first_frame examples/flf2v_input_first_frame.png --last_frame examples/flf2v_input_last_frame.png \
    --use_prompt_extend --prompt_extend_model /h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/model/Qwen2.5-VL-3B-Instruct \
    --prompt "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird’s feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
