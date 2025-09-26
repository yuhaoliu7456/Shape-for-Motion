from ruamel.yaml import YAML
import sys

def modify_config(base_config_path, new_config_path, new_params):
    # Initialize YAML with round trip mode
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.boolean_representation = ['False', 'True']
    
    # Read the base config
    with open(base_config_path, 'r') as file:
        config = yaml.load(file)
    
    # Update the parameters
    for key, value in new_params.items():
        config[key] = value
    
    # Write the new config
    with open(new_config_path, 'w') as file:
        yaml.dump(config, file)

base_config_path = sys.argv[1]
new_config_path = sys.argv[2]
new_params = {
    'source_path': sys.argv[3],
    'model_path': sys.argv[4],
    'train_json': sys.argv[5],
    'test_json': sys.argv[6]
}
modify_config(base_config_path, new_config_path, new_params)
