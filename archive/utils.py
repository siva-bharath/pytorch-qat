# full training report 
6348805e1f6699a0d5ed7fedc430646d25cceb89
num_epochs = 20
best_acc = 0.0
print("\nStarting fine-tuning...")

for epoch in range(1, num_epochs + 1):
    # Training
    train_acc = train_one_epoch(float_model, criterion, optimizer, data_loader, device, epoch, best_acc)
    
    # Validation
    float_model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, targets in data_loader_test:
            images = images.to(device)
            targets = targets.to(device)
            outputs = float_model(images)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    
    val_acc = 100.*val_correct/val_total
    print(f'\nEpoch {epoch}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': float_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_acc,
        }, os.path.join(saved_model_dir, 'best_model.pth'))
    
    # Set back to training mode
    float_model.train()
    
    # Clear memory after each epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()



def objective(trial):
    """
    Single Optuna trial that:
      • Builds a fresh MobileNetV2 classifier head
      • Samples optimiser / LR / weight-decay / dropout
      • Runs a short fine-tuning and returns val accuracy
    """
    # -----  hyper-parameters to search  ---------------------------------
    lr           = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    opt_name     = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    momentum     = trial.suggest_float("momentum", 0.7, 0.99) if opt_name == "SGD" else None
    dropout_p    = trial.suggest_float("dropout", 0.0, 0.5)
    epochs       = trial.suggest_int("finetune_epochs", 5, 50)
    # --------------------------------------------------------------------

    # ── fresh copy of the (already fused) base model ─────────────────────
    model = load_model(saved_model_dir + float_model_file, num_classes=102)
    model.fuse_model()
    model.to(device)

    #  Freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    #  Replace dropout with the sampled p
    model.classifier[0] = nn.Dropout(dropout_p)

    #  Pick optimiser
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(
            model.classifier.parameters(), lr=lr,
            momentum=momentum, weight_decay=weight_decay
        )
    else:  # AdamW
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(), lr=lr,
            weight_decay=weight_decay
        )

    #  Short training loop (smaller epoch count = faster search)
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, criterion, optimizer,
                        data_loader, device, epoch)
        # --- quick val pass ------------------------------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, targets in data_loader_test:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                correct += logits.argmax(1).eq(targets).sum().item()
                total   += targets.size(0)
        val_acc = 100. * correct / total
        best_val_acc = max(best_val_acc, val_acc)

        # report & prune
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # clear GPU RAM between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return best_val_acc

# Fine-tune the classifier
optimizer = torch.optim.SGD(
    float_model.classifier.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=0.01
)


