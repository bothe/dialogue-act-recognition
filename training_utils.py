from keras.callbacks import TensorBoard, ModelCheckpoint, RemoteMonitor, ReduceLROnPlateau


def keras_callbacks(con_model_name, logdir):
    tb_logging = TensorBoard(log_dir=logdir)
    ck_pts = ModelCheckpoint(con_model_name, monitor='val_loss', save_best_only=True)
    rmt_monitor = RemoteMonitor(root='http://localhost:9000', path='logs', field='data', headers=None,
                                send_as_json=False)
    rdc_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.0001,
                               cooldown=0, min_lr=0)
    cbs = [ck_pts, rdc_lr, rmt_monitor, tb_logging]
    return cbs
