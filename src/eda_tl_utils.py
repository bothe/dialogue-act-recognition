def merge_emotion_classes(utt_Emotion):
    emotion_classes = ['hap', 'neu', 'sad', 'fru', 'ang', 'xxx']
    utt_Emotion_merger_ids = []
    utt_Emotion_merged = utt_Emotion
    for i in range(len(utt_Emotion)):
        utt_Emotion_merged[i] = utt_Emotion[i]
        if utt_Emotion_merged[i] == "fea":
            utt_Emotion_merged[i] = "sad"
        if utt_Emotion_merged[i] == "sur":
            utt_Emotion_merged[i] = "hap"
        if utt_Emotion_merged[i] == "exc":
            utt_Emotion_merged[i] = "hap"
        if utt_Emotion_merged[i] == "dis":
            utt_Emotion_merged[i] = "sad"
        if utt_Emotion_merged[i] == "oth":
            utt_Emotion_merged[i] = "xxx"
        utt_Emotion_merger_ids.append(emotion_classes.index(utt_Emotion_merged[i]))

    return emotion_classes, utt_Emotion_merged, utt_Emotion_merger_ids
