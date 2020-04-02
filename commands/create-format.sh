# create directories
mkdir data
mkdir data/input
mkdir data/output
mkdir config
mkdir features
mkdir models
mkdir notebooks
mkdir logs
mkdir script
# download utils 
wget --no-check-certificate https://github.com/Soichiro-Fujioka/ds-format/archive/master.zip
# arange directories
unzip master.zip
rm master.zip
cp -r ds-format-master/* ./ 
rm -r ds-format-master
