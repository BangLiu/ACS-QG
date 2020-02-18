# -*- coding:utf-8 -*-
import visdom
import time
import numpy as np


class Visualizer(object):
    '''
    Some visdom operation and plot functions for visualization.
    We can use`self.vis.function` to use original visdom functions.
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """ To change visdom configuration.
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """ 一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)

    def plot(self, name, y, **kwargs):
        """ self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def log(self, info, win='log_text'):
        """ self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
