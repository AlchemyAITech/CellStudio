import torchvision.transforms as T

def build_transforms(aug_configs):
    """
    Dynamically builds a torchvision transform pipeline based on JSON array configuration.
    """
    if not aug_configs:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    transforms_list = [T.Resize((224, 224))]
    
    for aug in aug_configs:
        if isinstance(aug, dict):
            name = aug.get("name")
            kwargs = {k: v for k, v in aug.items() if k != "name"}
            if hasattr(T, name):
                transform_cls = getattr(T, name)
                transforms_list.append(transform_cls(**kwargs))
            else:
                print(f"[Warn] Augmentation '{name}' not found in torchvision.transforms")
                
    transforms_list.append(T.ToTensor())
    transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms_list)
