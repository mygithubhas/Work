
import tensorflow as tf
import sys
sys.path.append("../")
from train_models.MTCNN_config import config


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):#导入cnn结构和路径
        #create a graph
        graph = tf.Graph()#计算图
        with graph.as_default():#在默认计算图中执行
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
			#空占位符，输入
            self.width_op = tf.placeholder(tf.int32, name='image_width')
			#空占位符，图片宽度
            self.height_op = tf.placeholder(tf.int32, name='image_height')
			#空占位符，图片高度
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])#把输入图片格式化为1*height_op*width_op*3的矩阵
            #self.cls_prob batch*2
            #self.bbox_pred batch*4
            #construct model here
            #self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #contains landmark
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)#把矩阵传入网络
            
            #allow 
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
			#tf.ConfigProto()配置Session运行参数
            saver = tf.train.Saver()#保存模型
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
			#获得模型
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)
    def predict(self, databatch):
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        return cls_prob, bbox_pred
