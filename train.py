import loaders, models



omg = loaders.OmniglotDataset(dataset_info='./train_dataset_temp.txt')
print(len(omg))

omg = loaders.OmniglotDataset(dataset_info='./val_dataset_temp.txt')
print(len(omg))


siam = models.SiameseNetwork()

print(siam)