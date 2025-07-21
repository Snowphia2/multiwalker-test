# Install SUMO
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc

# Install MAPDN
cd packages/MAPDN/mapdn/environments/var_voltage_control/data
gdown https://drive.google.com/file/d/1-GGPBSolVjX1HseJVblNY3KoTqfblmLh/
view?usp=sharing --fuzzy
unzip voltage_control_data.zip
rm voltage_control_data.zip
mv voltage_control_data/* .
apt install freeglut3-dev python3-opengl -y