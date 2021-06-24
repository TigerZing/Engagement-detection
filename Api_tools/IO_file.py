import yaml
import io

# Read configuration file
def read_config(file_path):
    config_file = open(file_path)
    parsed_content = yaml.load(config_file, Loader=yaml.FullLoader)
    return parsed_content

# Test function
if __name__=="__main__":
    read_config("/home/tigerzing/Documents/source/new_code/config.yml")
