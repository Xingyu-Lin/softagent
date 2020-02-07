python chester/pull_s3_result.py PlaNet-0202 > tmp1.txt & disown
python chester/pull_s3_result.py PlaNet-0204-ClothFold > tmp2.txt & disown
python chester/pull_s3_result.py PlaNet-0205-ClothFlatten > tmp3.txt & disown

python chester/pull_s3_result.py RIG-128-0202-all > tmp4.txt & disown
python chester/pull_s3_result.py RIG-128-0204-ClothFold > tmp5.txt & disown

python chester/pull_s3_result.py model-free-key-point-0202 > tmp6.txt & disown
python chester/pull_s3_result.py model-free-key-point-0203-last-2-seeds > tmp7.txt & disown
python chester/pull_s3_result.py model-free-key-point-0204-ClothFold > tmp8.txt & disown
python chester/pull_s3_result.py model-free-key-point-0205-ClothFlatten > tmp9.txt & disown
