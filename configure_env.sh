# Install python dependencies
# this may take a while (~a few minutes)
# please edit pyproject.toml/[[tool.uv.index]] to use a index which is faster in your region; if you are not in China, you can remove the index line to use pypi.org
uv sync

# Install SUMO
apt update
apt install software-properties-common -y
add-apt-repository ppa:sumo/stable
apt-get update
apt-get install sumo sumo-tools sumo-doc -y
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc

# Install MAPDN
mkdir packages/MAPDN/mapdn/environments/var_voltage_control/data
cd packages/MAPDN/mapdn/environments/var_voltage_control/data
uv pip install gdown
uv run gdown https://drive.google.com/file/d/1-GGPBSolVjX1HseJVblNY3KoTqfblmLh/view?usp=sharing --fuzzy
unzip voltage_control_data.zip
rm voltage_control_data.zip
mv voltage_control_data/* .
apt install freeglut3-dev python3-opengl -y