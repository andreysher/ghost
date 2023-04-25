CUDA_VISIBLE_DEVICES=0 \
python inference.py \
--source_paths examples/images/elon_musk.jpg \
--target_video examples/videos/dirtydancing.mp4 \
--batch_size 1
