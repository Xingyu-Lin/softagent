# remove old clothfold results
for i in 26 27 28 29 30
do
    rm -rf data/yufei_s3_data/PlaNet-0202/*00$i
    rm -rf data/yufei_s3_data/RIG-128-0202-all/*00$i
done

for i in 25 26 27 28 29 30
do
    rm -rf data/yufei_s3_data/model-free-key-point-0202/*00$i
done

for i in 17 18 19 20
do
    rm -rf data/yufei_s3_data/model-free-key-point-0203-last-2-seeds/*00$i
done

# remove old clothflatten results
for i in 19 20 21 22 23 24
do 
    rm -rf data/yufei_s3_data/model-free-key-point-0202/*00$i
done

for i in 16 17 18 19 20
do 
    rm -rf data/yufei_s3_data/PlaNet-0202/*00$i
done

for i in 13 14 15 16 
do
    rm -rf data/yufei_s3_data/model-free-key-point-0203-last-2-seeds/*00$i
done

rm tmp*.txt