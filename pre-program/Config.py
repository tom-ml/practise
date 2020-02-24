class Config(object):
    file_name = '../dataset/remove_stopwords-10line.csv'
    weight_file = 'hotels_model.h5'
    # 根据前六个字预测第七个字
    max_len = 6
    batch_size = 512
    learning_rate = 0.001