# it creates a CSV file with the paths of the images and their labels
import csv

num_users = 55
num_signatures = 24
base_path_org = "signatures/full_org/forgeries_user"
base_path_forg = "signatures/full_forg/forgeries_user"

with open('signatures_dataset.csv', 'w', newline='') as csvfile:
    fieldnames = ['user_id', 'image_path', 'label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for user_id in range(1, num_users + 1):
        for signature_id in range(1, num_signatures + 1):
            # Original signature
            org_path = f"{base_path_org.replace('user', str(user_id))}_{signature_id}.png"
            writer.writerow({'user_id': user_id, 'image_path': org_path, 'label': 'true'})
            
            # Forged signature
            forg_path = f"{base_path_forg.replace('user', str(user_id))}_{signature_id}.png"
            writer.writerow({'user_id': user_id, 'image_path': forg_path, 'label': 'forged'})
