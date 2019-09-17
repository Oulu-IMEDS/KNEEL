wget http://mipt-ml.oulu.fi/models/KNEEL/snapshots_release.tar.xz
tar -xvf snapshots_release.tar.xz

docker run -it --name landmark_inference --rm \
            -v ${pwd}:/workdir/ \
            -v snapshots_release:/snapshots/:ro \
            -v OKOA:/data/:ro --ipc=host \
            kneel_inference python -u inference_new_data.py \
            --dataset_path ${HOME} \
            --dataset_name OKOA \
            --workdir ${HOME} \
            --mean_std_path /snapshots/mean_std.npy\
            --lc_snapshot_path /snapshots/lext-devbox_2019_07_14_16_04_41 \
            --hc_snapshot_path /snapshots/lext-devbox_2019_07_14_16_04_41 \
            --device cpu \
            --refine True
