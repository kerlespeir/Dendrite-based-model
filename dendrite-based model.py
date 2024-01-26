import neuron
from neuron import h
from neuron.units import ms,mV,μm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import numpy as np
h.load_file("stdrun.hoc")

class Cell:
    def __init__(self,name, gid, x, y, z, theta):
        self._gid = gid
        self._name = name
        self._setup_morphology()
        self.all = self.soma.wholetree()
        self._setup_biophysics()
        self.x = self.y = self.z = 0  # <-- NEW
        h.define_shape()
        self._rotate_z(theta)  # <-- NEW
        self._set_position(x, y, z)  # <-- NEW
        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)

    def __repr__(self):
        return "{}[{}]".format(self._name, self._gid)

    # everything below here is NEW

    def _set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                )
        self.x, self.y, self.z = x, y, z

    def _rotate_z(self, theta):
        """Rotate the cell about the Z axis."""
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))

class Neuron_Cell(Cell):
    def __init__(self,name, gid, x, y, z, theta):
        super().__init__(name, gid, x, y, z, theta)

    def _setup_morphology(self):
        self.soma = h.Section(name="soma",cell=self)
        self.dendrite = h.Section(name="dendrite",cell=self)
        self.dendrite.connect(self.soma)
        self.dendrite.nseg = 20

        self.soma.L = self.soma.diam = 12.6157 * μm
        self.dendrite.L = 200 * μm
        self.dendrite.diam = 1 * µm
        # define spike input
        inp = (np.power(np.random.rand(200),3)-0.4)/20
        self.pvec = h.Vector().from_python(inp)
        self.iclamp = h.IClamp(self.soma(0.5))
        self.iclamp.dur = 1e9
        self.pvec.play(self.iclamp, self.iclamp._ref_amp, True)
    def _setup_biophysics(self):
        #set soma membrane resistance and capacitance
        for section in self.all:
            section.Ra=100
            section.cm=1
        #set soma channel parameters in hh model
        self.soma.insert("hh1")
        for segment in self.soma:
            segment.hh1.gnabar = 0.12
            segment.hh1.gkbar = 0.036
            segment.hh1.gl = 0.0003
            segment.hh1.el = -54.3 * mV
        #Insert passive dendrite 
        self.dendrite.insert("pas")
        for segment in self.dendrite:
            segment.pas.g = 0.001 #passive conductance
            segment.pas.e = -66 * mV
    def initial_spike(self,dur,amp):
        self.iclamp = h.IClamp(self.soma(0.5))
        self.iclamp.delay=0
        self.iclamp.dur=dur
        self.iclamp.amp=amp
    def create_n_BallAndStick(name, n, d, start_x_y:tuple,E_cell=True):
        """create n*n cells on a plane, the distance between two Adjacent cells is d
            return a 1D list"""
        cells = []
        for i in range(n):
            cells.append(list())
            for j in range(n):
                if E_cell==False:
                    if (i%3==0 and j%3==0) or (i%3==1 and j%3==2) or (i%3==2 and j%3==1):
                        cells[i].append(Neuron_Cell(name, (i,j),i*d+start_x_y[0] ,j*d+start_x_y[1] , 0, 0))
                    else:
                        cells[i].append(None)
                else:
                    cells[i].append(Neuron_Cell(name, (i,j),i*d+start_x_y[0] ,j*d+start_x_y[1] , 0, 0))
        return cells

#h.topology()

def dist(position1:tuple,position2:tuple):
    '''the dist function return the Euclidean distance of two neurons
    it consider the plane as a ring
    Attention: the input should be like (i,j),where i and j are index. Not physical distance'''
    dx = np.abs(position1[0]-position2[0])
    dy = np.abs(position1[1]-position2[1])
    if dx>En/2:
        dx = En-dx
    else:pass
    if dy>En/2:
        dy = En-dy
    else:pass
    dist = np.sqrt(np.square(dx)+np.square(dy))
    return dist

def set_connect(file_name,Weo=0.01,Wio=-0.002,Ce=2e-1,Ci=2e-2,dis_I_max=0.5,dis_I_bias=0.1):
    # set excitatory connection in cells, i->j
    for cell_i in [elem for elements in E_cells for elem in elements]:
        for cell_j in [elem for elements in E_cells for elem in elements]:
            if(cell_i==cell_j):continue
            dis = np.square(dist(cell_i._gid,cell_j._gid))*Ce
            if dis>1: continue
            syn = h.ExpSyn(cell_j.dendrite(dis))
            nc = h.NetCon(cell_i.soma(0.5)._ref_v, syn, sec=cell_i.soma)
            nc.weight[0] = Weo
            nc.delay = 1 * ms
            
            syn_list.append(syn)
            nc_list.append(nc)
    for cell_i in [elem for elements in E_cells for elem in elements]:
        for cell_j in [elem for elements in I_cells for elem in elements]:
            if cell_j==None:
                continue
            dis = np.square(dist(cell_i._gid,cell_j._gid))*Ce
            if dis>1: continue
            
            syn = h.ExpSyn(cell_j.dendrite(dis))
            
            #print(f"connection:{cell_i._gid},{cell_j._gid},dis:{dis},weight:{Weo}")
            nc = h.NetCon(cell_i.soma(0.5)._ref_v, syn, sec=cell_i.soma)
            nc.weight[0] = Weo
            nc.delay = 1 * ms
            
            syn_list.append(syn)
            nc_list.append(nc)
                    
    # set inhibitrory connection in cells, i->j
    for cell_i in [elem for elements in I_cells for elem in elements]:
        for cell_j in [elem for elements in E_cells for elem in elements]:
            if cell_i==None:
                continue
            if(cell_i==cell_j):continue
            
            dis = np.square(dist(cell_i._gid,cell_j._gid))*Ci  # inhibitory synap
            if dis>=1: continue
            
            syn = h.ExpSyn(cell_j.soma(dis))
            
            #print(f"connection:{cell_i._gid},{cell_j._gid},dis:{dis},weight:{Weo}")
            nc = h.NetCon(cell_i.soma(0.5)._ref_v, syn, sec=cell_i.soma)
            nc.weight[0] = Wio
            nc.delay = 1 * ms
            
            syn_list.append(syn)
            nc_list.append(nc)
    for cell_i in [elem for elements in I_cells for elem in elements]:
        for cell_j in [elem for elements in I_cells for elem in elements]:
            if cell_i==None or cell_j==None:
                continue
            if(cell_i==cell_j):continue
            
            dis = np.square(dist(cell_i._gid,cell_j._gid))*Ci  # inhibitory synap
            if dis>1: continue
            
            syn = h.ExpSyn(cell_j.soma(dis))
            
            #print(f"connection:{cell_i._gid},{cell_j._gid},dis:{dis},weight:{Weo}")
            nc = h.NetCon(cell_i.soma(0.5)._ref_v, syn, sec=cell_i.soma)
            nc.weight[0] = Wio
            nc.delay = 1 * ms
            
            syn_list.append(syn)
            nc_list.append(nc)
    file_name+=file_name+f'Weo{Weo}_Wio{Wio}'
    return file_name
    
def array_dist(a , b):
    if (a-b)>En/2:
        return a-b-En
    elif a-b<-En/2:
        return a-b+En
    else: return a-b

def calcu_velocity():
    theta_v = list()
    for i in range(En):
        for j in range(En):
            for spike_time in list(E_cells[i][j].spike_times):
                min_cell=None
                t0=5
                if spike_time<time*2/3:continue
                for m in range((i-2)%En,(i+3)%En):
                    for n in range((j-2)%En,(j+3)%En):
                        for st in list(E_cells[m][n].spike_times):
                            if spike_time-st>5 or (m==i and n==j) or spike_time-st<2:continue
                            if min_cell==None or dist(E_cells[i][j]._gid,E_cells[m][n]._gid)<dist(E_cells[i][j]._gid,min_cell._gid):
                                min_cell=E_cells[m][n]
                                t0=spike_time-st
                if min_cell:
                    theta_v.append((array_dist(E_cells[i][j]._gid[0],min_cell._gid[0]),array_dist(E_cells[i][j]._gid[1],min_cell._gid[1])))
                
    return np.array(theta_v)    
   
result = list()
for i in range(5):
    print(f'Doing {i}/10')
    fai_list = list()
    for t in range(20):
        syn_list = list()
        nc_list = list()    

        file_name=''
        time=200
        En = 24  # En*En is E_cells numbers in the plane
        In = 24  # In*In is I_cells numbers in the plane
        file_name+=f'En{En}_In{In/3}'
        E_cells = Neuron_Cell.create_n_BallAndStick('E_cells',En, 20, (0,0))
        I_cells = Neuron_Cell.create_n_BallAndStick('I_cells',In, 20, (10,10),E_cell=False)

        file_name = set_connect(file_name,Weo=0.014+0.001*i,Wio=-0.002,Ce=2e-1,Ci=2e-2)
        t = h.Vector().record(h._ref_t)
        soma_v_center = h.Vector().record(E_cells[4][4].soma(0.5)._ref_v)
        soma_v_list = list()
        for cell in [elem for elements in E_cells for elem in elements]:
            soma_v_list.append(h.Vector().record(cell.soma(0.5)._ref_v))

        h.finitialize(-65 * mV)
        h.continuerun(time * ms)
        v=calcu_velocity()
        fai = np.linalg.norm(v.mean(0))/np.array(np.linalg.norm(v,axis=1)).mean()
        fai_list.append(fai)
        '''    # 创建一个空白图表
        fig, ax = plt.subplots()
        # 设置初始数据，假设是一个 10x10 的矩阵
        data = np.zeros((En,En))
        # 创建热力图对象
        heatmap = ax.imshow(data, cmap='bwr', interpolation='nearest', vmin=-100, vmax=50)
        plt.colorbar(heatmap)  # 添加颜色条
        # 更新函数，用于每一帧的更新
        def update(frame):
            # 在每一帧更新数据
            for i,cell in enumerate([elem for elements in E_cells for elem in elements]):
                data[cell._gid[0]][cell._gid[1]]=soma_v_list[i][frame]
            heatmap.set_array(data)
            return heatmap,
        # 创建动画对象
        animation = FuncAnimation(fig, update, frames=time*25, interval=0.01, blit=True)
        # 显示动画
        #animation.save(f'all_soma{file_name}animation.gif', writer=PillowWriter(fps=30))
        fig.show()

        plt.figure()
        for i, cell in enumerate([elem for elements in E_cells for elem in elements]):
            plt.vlines(list(cell.spike_times), i + 0.5, i + 1.5)
        #plt.savefig(f'all_soma{file_name}.vlines.png')
        plt.show()
        plt.clf()'''

    print(np.array(fai_list).mean())
    result.append(np.array(fai_list).mean())


print('finish')
exit()
