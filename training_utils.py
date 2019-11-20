from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def keras_callbacks(con_model_name, logdir):
    tb_logging = TensorBoard(log_dir=logdir)
    ck_pts = ModelCheckpoint(con_model_name, monitor='val_loss', save_best_only=True)
    rdc_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0,
                               mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    early_stopper = EarlyStopping(patience=10)
    cbs = [ck_pts, rdc_lr, tb_logging, early_stopper]
    return cbs
