import pycolmap

temp = ["60", "80", "120", "140", "280"]
imgs = []
for t in temp:
    with open(f"../TexturePoorSfM_dataset/1002/5bag000/images/{t}") as img:
        imgs.append(img)

r = pycolmap.Reconstruction("../TexturePoorSfM_dataset/1002/5bag000/images")
r.
