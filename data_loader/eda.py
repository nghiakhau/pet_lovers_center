import numpy as np
import pandas as pd
import cv2
import glob
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import partial_corr


def get_image(filepath, img_size=224):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(
        src=img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def get_transformed_image(filepath, img_size=224):
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=45),
                    transforms.ToTensor()
                    ])
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(
        src=img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    img = transform(img)
    img = np.moveaxis(img.numpy(), 0, -1)
    return img


def get_images(field_name, img_size=224, figsize=(25, 15), rand_state=10):
    imgs1 = data[data[field_name] == 0].sample(
        5, random_state=rand_state).Id.values
    imgs2 = data[data[field_name] == 1].sample(
        5, random_state=rand_state).Id.values

    imgs = np.concatenate([imgs1, imgs2])
    imgs = [get_image(imgname, img_size) for imgname in imgs]

    _, ax = plt.subplots(2, 5, figsize=figsize, sharex=True, sharey=True)

    ax[0, 0].set_title(field_name + " == 0")
    ax[1, 0].set_title("\n" + field_name + " == 1")
    for i in range(2):
        for j in range(5):
            ax[i, j].imshow(imgs[i*5 + j])
    plt.show()


# select random img to show
img = np.random.choice(glob.glob('data/petfinder-pawpularity-score/train/*'))
plt.imshow(get_image(img))
plt.imshow(get_transformed_image(img))

data_dir = "data/petfinder-pawpularity-score/train.csv"
data = pd.read_csv(data_dir)

# data['sum_feature'] = (data['Subject Focus'] + data['Eyes'] + data['Face'] +
#                        data['Near'] + data['Action'] + data['Accessory'] +
#                        data['Group'] + data['Collage'] + data['Human'] +
#                        data['Occlusion'] + data['Info'] + data['Blur'])

corr = data.corr()['Pawpularity'].sort_values()
corr
# partial corr
pcorr = data.pcorr()['Pawpularity'].round(3).sort_values()
pcorr

data['Id'] = data['Id'].apply(
    lambda x: "data/petfinder-pawpularity-score/train/" + x + ".jpg")
data.head()

print(data.shape)
288 + 369
len(data[data['Pawpularity'] == 100])
len(data[((data['Pawpularity'] >= 10) & (data['Pawpularity'] != 100))])
clean_data = data[((data['Pawpularity'] >= 10) & (data['Pawpularity'] != 100))]
clean_data.head(10)
clean_data.shape
clean_data.tail()
clean_data.reset_index(drop=True, inplace=True)
clean_data.tail()

# Pawpularity score distribution
# There are 288 observations that have Pawpularity score = 100 (propably noise)
targ_cts = data.Pawpularity.value_counts()
fig = plt.figure(figsize=(20, 10))
sns.barplot(x=targ_cts.sort_values(ascending=False).index,
            y=targ_cts.sort_values(ascending=False).values,
            palette='summer')
plt.title('Target Distribution')
plt.show()


data['score_bin'] = data.Pawpularity.apply(lambda x: int(x/10))
bin_cts = data.score_bin.value_counts()
fig = plt.figure(figsize=(20, 10))
sns.barplot(x=bin_cts.sort_values(ascending=False).index,
            y=bin_cts.sort_values(ascending=False).values,
            palette='summer')
plt.title('Score Bins Distribution')
plt.show()


SEED = 2000
FIGSIZE = (25, 10)
IMG_SIZE = 500


FEATURE = 'Blur'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Subject Focus'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Eyes'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Info'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Action'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Near'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Collage'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
# digitally retouched photo
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Occlusion'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
# there is something you want to see, but can't due to some reasons
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Human'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Face'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Accessory'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()

FEATURE = 'Group'
covar = list(data.columns)
covar.remove('Id')
covar.remove('Pawpularity')
covar.remove('score_bin')
covar.remove(FEATURE)
partial_corr(data, 'Pawpularity', FEATURE, covar=covar)
get_images(FEATURE, IMG_SIZE, FIGSIZE, SEED)
fig = plt.figure(figsize=(20, 10))
sns.countplot(data=data, x='score_bin', hue=FEATURE)
plt.title(FEATURE + " in Popularity Bins")
plt.show()
