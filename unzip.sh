#!/bin/bash

# 指定包含 ZIP 文件的目录
source_directory="data/umi_data/zip"

# 目标解压路径
destination="data/umi_data/zarr"

# 确保目标路径存在
mkdir -p "$destination"

# 遍历指定目录下的所有 ZIP 文件并解压到同名文件夹
for dataset in "$source_directory"/*.zip; do
	    if [ -f "$dataset" ]; then
		            folder_name="$destination/$(basename "$dataset" .zip)"
			            mkdir -p "$folder_name"
				            echo "解压: $dataset 到 $folder_name"
					            unzip -o "$dataset" -d "$folder_name"
						        fi
						done

						echo "所有 ZIP 数据集解压完成！"
