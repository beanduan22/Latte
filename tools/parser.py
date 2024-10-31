def parse_model_path(model_name):
    parts = model_name.split('_')
    dataset = parts[0]
    model = '_'.join(parts[1:])
    path = f"./weight/{dataset}/{model}"
    return path

def parse_model_instance(model_name):
    parts = model_name.split('_')
    dataset = parts[0]
    model = parts[1]
    Model = f"{model.capitalize()}_Model_{dataset}()"
    vqvae = f"{model.capitalize()}_VQVAE_{dataset}()"
    xd = f"ImageDiscriminator()"
    zd = f"LatentDiscriminator()"
    return Model, vqvae, xd, zd