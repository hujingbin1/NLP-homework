export LANG=en_US.UTF-8
python process.py
nohup python main.py 2>> >(echo $(date) >>.\\output\\error\\error.log && cat >> .\\output\\error\\error.log) >> >(echo $(date) >>.\\output\\train\\train.log && cat >> .\\output\\train\\train.log) &
nohup python main.py &> >(echo $(date) > .\\output\\result\\result.log && cat >> .\\output\\result\\result.log) &
