

models = ["Ford", "Ford", "Mazda", "Mazda", "Mazda"]

models_id = {}

i = 0
for model in models:
    if model not in models_id:
        models_id[model] = i
        i += 1

print(models_id)
