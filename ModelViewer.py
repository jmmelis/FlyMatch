from __future__ import print_function
import sys
#import vtk
#from vtk.numpy_interface import dataset_adapter as dsa
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidget, QTableWidgetItem, QVBoxLayout, QFileDialog
#from geomdl import BSpline
#from geomdl import utilities
#from geomdl import exchange
#from geomdl import NURBS
#from geomdl import tessellate
#from geomdl.utilities import make_triangle_mesh
import numpy as np
import numpy.matlib
import os
import os.path
from os import path
import copy
import time
import json
import h5py
import cv2

from state_fitter import StateFitter

#from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from session_select import Ui_Session_Dialog

class CheckableDirModel(QtGui.QDirModel):
	def __init__(self, parent=None):
		QtGui.QDirModel.__init__(self, None)
		self.checks = {}

	def data(self, index, role=QtCore.Qt.DisplayRole):
		if role != QtCore.Qt.CheckStateRole:
			return QtGui.QDirModel.data(self, index, role)
		else:
			if index.column() == 0:
				return self.checkState(index)

	def flags(self, index):
		return QtGui.QDirModel.flags(self, index) | QtCore.Qt.ItemIsUserCheckable

	def checkState(self, index):
		if index in self.checks:
			return self.checks[index]
		else:
			return QtCore.Qt.Unchecked

	def setData(self, index, value, role):
		if (role == QtCore.Qt.CheckStateRole and index.column() == 0):
			self.checks[index] = value
			self.emit(QtCore.SIGNAL("dataChanged(QModelIndex,QModelIndex)"), index, index)
			return True 

		return QtGui.QDirModel.setData(self, index, value, role)

# QDialog clasess:

class SelectFolderWindow(QtGui.QDialog, Ui_Session_Dialog):

	def __init__(self, directory, parent=None):
		super(SelectFolderWindow,self).__init__(parent)
		self.setupUi(self)
		self.folder_name = None
		self.folder_path = None
		self.file_model = QFileSystemModel()
		self.directory = directory
		self.file_model.setRootPath(directory)
		self.folder_tree.setModel(self.file_model)
		self.folder_tree.setRootIndex(self.file_model.index(self.directory));
		self.folder_tree.clicked.connect(self.set_session_folder)

	def update_file_model(self,new_dir):
		self.directory = new_dir
		self.file_model.setRootPath(new_dir)

	def set_session_folder(self, index):
		indexItem = self.file_model.index(index.row(), 0, index.parent())
		self.folder_name = self.file_model.fileName(indexItem)
		self.folder_path = self.file_model.filePath(indexItem)
		self.selected_session.setText(self.folder_path)

class Graph(pg.GraphItem):
	def __init__(self,graph_nr):
		self.graph_nr = graph_nr
		self.dragPoint = None
		self.dragOffset = None
		self.textItems = []
		pg.GraphItem.__init__(self)
		self.scatter.sigClicked.connect(self.clicked)
		self.onMouseDragCb = None
		
	def setData(self, **kwds):
		self.text = kwds.pop('text', [])
		self.data = copy.deepcopy(kwds)
		
		if 'pos' in self.data:
			npts = self.data['pos'].shape[0]
			self.data['data'] = np.empty(npts, dtype=[('index', int)])
			self.data['data']['index'] = np.arange(npts)
		self.setTexts(self.text,self.data)
		self.updateGraph()
		
	def setTexts(self, text, data):
		for i in self.textItems:
			i.scene().removeItem(i)
		self.textItems = []
		#for t in text:
		for i,t in enumerate(text):
			item = pg.TextItem(t)
			if len(data.keys())>0:
				item.setColor(data['textcolor'][i])
			self.textItems.append(item)
			item.setParentItem(self)
		
	def updateGraph(self):
		pg.GraphItem.setData(self, **self.data)
		for i,item in enumerate(self.textItems):
			item.setPos(*self.data['pos'][i])

	def setOnMouseDragCallback(self, callback):
		self.onMouseDragCb = callback
		
	def mouseDragEvent(self, ev):
		if ev.button() != QtCore.Qt.LeftButton:
			ev.ignore()
			return
		
		if ev.isStart():
			# We are already one step into the drag.
			# Find the point(s) at the mouse cursor when the button was first 
			# pressed:
			pos = ev.buttonDownPos()
			pts = self.scatter.pointsAt(pos)
			if len(pts) == 0:
				ev.ignore()
				return
			self.dragPoint = pts[0]
			ind = pts[0].data()[0]
			self.dragOffset = self.data['pos'][ind] - pos
		elif ev.isFinish():
			self.dragPoint = None
			return
		else:
			if self.dragPoint is None:
				ev.ignore()
				return
		
		ind = self.dragPoint.data()[0]
		self.data['pos'][ind] = ev.pos() + self.dragOffset
		self.updateGraph()
		ev.accept()
		if self.onMouseDragCb:
			PosData = self.data['pos'][ind]
			PosData = np.append(PosData,ind)
			PosData = np.append(PosData,self.graph_nr)
			self.onMouseDragCb(PosData)
		
	def clicked(self, pts):
		print("clicked: %s" % pts)

class ModelViewer(pg.GraphicsWindow):

	def __init__(self, parent=None):
		pg.GraphicsWindow.__init__(self)
		self.setParent(parent)

		self.w_sub = self.addLayout(row=0,col=0)

		self.v_list = []
		self.img_list = []
		self.frame_list = []

		# Parameters:
		self.frame_nr = 0

		self.N_cam = 3

		self.cam_folders = ['cam_1','cam_2','cam_3']

		self.frame_name = 'frame_'

		self.graph_list = []

		#self.key_pts_L = np.array([
		#	[0.0, 0.0, 0.0, 1.0],
		#	[0.3253, 2.3205, 0.0, 1.0],
		#	[0.0, 2.6241, 0.0, 1.0],
		#	[-0.2386, 2.5591, 0.0, 1.0],
		#	[-0.7012, 1.5976, 0.0, 1.0],
		#	[-0.7880, 0.8892, 0.0, 1.0]])

		self.key_pts_L = np.array([
			[0.0, 0.0, 0.0, 1.0],
			[0.0, 2.6241, 0.0, 1.0]])

		self.wing_L_pts = np.array([
			[0.0, 0.0, 0.0, 1.0],
			[0.0, 2.6241, 0.0, 1.0]])

		self.key_pts_R = np.array([
			[0.0, 0.0, 0.0, 1.0],
			[0.0, -2.6241, 0.0, 1.0]])

		self.wing_R_pts = np.array([
			[0.0, 0.0, 0.0, 1.0],
			[0.0, -2.6241, 0.0, 1.0]])

		self.wing_L_uv = []
		self.wing_R_uv = []

		self.state_calc = StateFitter()

		self.state_L = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.state_R = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

		self.scale_L = 1.0
		self.scale_R = 1.0

		self.drag_lines = True

		self.cnt_lines = False

		self.contours_L = []
		self.contours_R = []

		self.gain = 2.0

		self.ry_L = 0.0

		self.ry_R = 0.0

	def setFrameNR(self,frame_nr):
		self.frame_nr = frame_nr

	def set_session_folder(self,session_folder):
		self.session_folder = session_folder

	def set_output_folder(self):
		self.output_folder = self.session_folder + '/manual_tracking'

	def set_mov_folder(self,mov_folder):
		self.mov_folder = mov_folder

	def load_frame(self,frame_nr):
		frame_list = []
		for i in range(self.N_cam):
			os.chdir(self.session_folder+'/'+self.mov_folder)
			os.chdir(self.cam_folders[i])
			img_cv = cv2.imread(self.frame_name + str(frame_nr) +'.bmp',0)
			frame_list.append(img_cv/255.0)
		self.set_output_folder()
		return frame_list

	def add_frame(self,frame_nr):
		self.frame_nr = frame_nr
		frame_list = self.load_frame(frame_nr)
		print(frame_list[0].shape)
		for i, frame in enumerate(frame_list):
			self.frame_list.append(frame)
			self.v_list.append(self.w_sub.addViewBox(row=1,col=i,lockAspect=True))
			frame_in = np.transpose(np.flipud(frame))
			self.img_list.append(pg.ImageItem(frame_in))
			self.v_list[i].addItem(self.img_list[i])
			self.v_list[i].disableAutoRange('xy')
			self.v_list[i].autoRange()

	def update_frame(self,frame_nr):
		self.frame_nr = frame_nr
		frame_list = self.load_frame(frame_nr)
		for i, frame in enumerate(frame_list):
			frame_in = np.transpose(np.flipud(frame))
			self.img_list[i].setImage(frame_in)

	def set_calibration_loc(self,calib_file,ds,cam_mag,cam_dist):
		self.calib_file = calib_file
		self.ds = ds
		self.cam_mag = cam_mag
		self.cam_dist = cam_dist
		self.LoadCalibrationMatrix(calib_file)
		self.CalculateProjectionMatrixes(self.ds,self.cam_mag)

	def LoadCalibrationMatrix(self,calib_file):
		self.calib_file = calib_file
		self.c_params = np.loadtxt(calib_file, delimiter='\t')
		self.N_cam = self.c_params.shape[1]

	def CalculateProjectionMatrixes(self,pix_size,magnification):
		self.img_size = []
		self.w2c_matrices = []
		self.c2w_matrices = []
		self.uv_shift = []
		for i in range(self.N_cam):
			self.img_size.append((int(self.c_params[13,i]),int(self.c_params[12,i])))
			# Calculate world 2 camera transform:
			C = np.array([[self.c_params[0,i], self.c_params[2,i], 0.0, 0.0],
				[0.0, self.c_params[1,i], 0.0, 0.0],
				[0.0, 0.0, 0.0, 1.0]])
			q0 = self.c_params[3,i]
			q1 = self.c_params[4,i]
			q2 = self.c_params[5,i]
			q3 = self.c_params[6,i]
			R = np.array([[2.0*pow(q0,2)-1.0+2.0*pow(q1,2), 2.0*q1*q2+2.0*q0*q3,  2.0*q1*q3-2.0*q0*q2],
				[2.0*q1*q2-2.0*q0*q3, 2.0*pow(q0,2)-1.0+2.0*pow(q2,2), 2.0*q2*q3+2.0*q0*q1],
				[2.0*q1*q3+2.0*q0*q2, 2.0*q2*q3-2.0*q0*q1, 2.0*pow(q0,2)-1.0+2.0*pow(q3,2)]])
			T = np.array([self.c_params[7,i],self.c_params[8,i],self.c_params[9,i]])
			K = np.array([[R[0,0], R[0,1], R[0,2], T[0]],
				[R[1,0], R[1,1], R[1,2], T[1]],
				[R[2,0], R[2,1], R[2,2], T[2]],
				[0.0, 0.0, 0.0, 1.0]])
			W2C_mat = np.dot(C,K)
			C2W_mat = np.dot(np.linalg.inv(K),np.linalg.pinv(C))
			self.w2c_matrices.append(W2C_mat)
			self.c2w_matrices.append(C2W_mat)
			uv_trans = np.zeros((3,1))
			if self.c_params.shape[0] ==  16:
				uv_trans[0] = (self.c_params[11,i]-self.c_params[13,i])/2.0-self.c_params[14,i]
				uv_trans[1] = (self.c_params[10,i]-self.c_params[12,i])/2.0-self.c_params[15,i]
			else:
				uv_trans[0] = (self.c_params[11,i]-self.c_params[13,i])/2.0
				uv_trans[1] = (self.c_params[10,i]-self.c_params[12,i])/2.0
			self.uv_shift.append(uv_trans)

	def set_u_cam_1_spin(self,spin_in):
		self.u_cam_1_spin = spin_in
		self.u_cam_1_spin.setMinimum(-10)
		self.u_cam_1_spin.setMaximum(10)
		self.u_cam_1_spin.setValue(0)
		self.u_cam_1_spin.valueChanged.connect(self.set_u_shift_cam_1)

	def set_u_shift_cam_1(self,u_in):
		self.uv_shift[0][0] -= u_in
		self.add_wing_contours()
		self.update_graphs()

	def set_v_cam_1_spin(self,spin_in):
		self.v_cam_1_spin = spin_in
		self.v_cam_1_spin.setMinimum(-10)
		self.v_cam_1_spin.setMaximum(10)
		self.v_cam_1_spin.setValue(0)
		self.v_cam_1_spin.valueChanged.connect(self.set_v_shift_cam_1)

	def set_v_shift_cam_1(self,v_in):
		self.uv_shift[0][1] -= v_in
		self.add_wing_contours()
		self.update_graphs()

	def set_u_cam_2_spin(self,spin_in):
		self.u_cam_2_spin = spin_in
		self.u_cam_2_spin.setMinimum(-10)
		self.u_cam_2_spin.setMaximum(10)
		self.u_cam_2_spin.setValue(0)
		self.u_cam_2_spin.valueChanged.connect(self.set_u_shift_cam_2)

	def set_u_shift_cam_2(self,u_in):
		self.uv_shift[1][0] -= u_in
		self.add_wing_contours()
		self.update_graphs()

	def set_v_cam_2_spin(self,spin_in):
		self.v_cam_2_spin = spin_in
		self.v_cam_2_spin.setMinimum(-10)
		self.v_cam_2_spin.setMaximum(10)
		self.v_cam_2_spin.setValue(0)
		self.v_cam_2_spin.valueChanged.connect(self.set_v_shift_cam_2)

	def set_v_shift_cam_2(self,v_in):
		self.uv_shift[1][1] -= v_in
		self.add_wing_contours()
		self.update_graphs()

	def set_u_cam_3_spin(self,spin_in):
		self.u_cam_3_spin = spin_in
		self.u_cam_3_spin.setMinimum(-10)
		self.u_cam_3_spin.setMaximum(10)
		self.u_cam_3_spin.setValue(0)
		self.u_cam_3_spin.valueChanged.connect(self.set_u_shift_cam_3)

	def set_u_shift_cam_3(self,u_in):
		self.uv_shift[2][0] -= u_in
		self.add_wing_contours()
		self.update_graphs()

	def set_v_cam_3_spin(self,spin_in):
		self.v_cam_3_spin = spin_in
		self.v_cam_3_spin.setMinimum(-10)
		self.v_cam_3_spin.setMaximum(10)
		self.v_cam_3_spin.setValue(0)
		self.v_cam_3_spin.valueChanged.connect(self.set_v_shift_cam_3)

	def set_v_shift_cam_3(self,v_in):
		self.uv_shift[2][1] -= v_in
		self.add_wing_contours()
		self.update_graphs()

	def save_uv_shift(self):
		calib_mat = np.zeros((16,3))
		calib_mat[0:14,:] = self.c_params[0:14,:]
		for n in range(self.N_cam):
			calib_mat[14,n] = -1.0*(self.uv_shift[n][0]-(self.c_params[11,n]-self.c_params[13,n])/2.0)
			calib_mat[15,n] = -1.0*(self.uv_shift[n][1]-(self.c_params[10,n]-self.c_params[12,n])/2.0)
		np.savetxt(self.calib_file,calib_mat,delimiter='\t')
		print('saved calib matrix:')
		print(calib_mat)

	def project2uv(self):
		self.wing_L_uv = []
		self.wing_R_uv = []
		for n in range(self.N_cam):
			uv_L_pts = np.dot(self.w2c_matrices[n],np.transpose(self.wing_L_pts))-self.uv_shift[n]
			uv_L_pts[1,:] = self.c_params[12,n]-uv_L_pts[1,:]
			uv_R_pts = np.dot(self.w2c_matrices[n],np.transpose(self.wing_R_pts))-self.uv_shift[n]
			uv_R_pts[1,:] = self.c_params[12,n]-uv_R_pts[1,:]
			self.wing_L_uv.append(uv_L_pts[0:2,:])
			self.wing_R_uv.append(uv_R_pts[0:2,:])

	def add_graphs(self):
		self.graph_list = []
		self.project2uv()
		for i in range(self.N_cam):
			self.graph_list.append(Graph(i))
			self.v_list[i].addItem(self.graph_list[i])
			self.wing_txt = ['L_r','L_t','R_r','R_t']
			self.wing_sym = ["o","o","o","o"]
			self.wing_clr = ['r','r','b','b']
			uv_pos = np.concatenate((np.transpose(self.wing_L_uv[i]),np.transpose(self.wing_R_uv[i])),axis=0)
			self.graph_list[i].setData(pos=uv_pos, size=2, symbol=self.wing_sym, pxMode=False, text=self.wing_txt, textcolor=self.wing_clr)
		self.add_wing_contours()

	def update_graphs(self):
		# Update keypoints:
		M_now_L = self.state_transform_L()
		key_pts_L0 = np.transpose(self.key_pts_L)
		self.wing_L_pts[0:3,:] = np.transpose(np.dot(M_now_L[0],key_pts_L0))
		#key_pts_L1 = np.transpose(self.key_pts_L[3,:])
		#self.wing_L_pts[3,:] = np.transpose(np.dot(M_now_L[1],key_pts_L1))
		#key_pts_L2 = np.transpose(self.key_pts_L[4,:])
		#self.wing_L_pts[4,:] = np.transpose(np.dot(M_now_L[2],key_pts_L2))
		#key_pts_L3 = np.transpose(self.key_pts_L[5,:])
		#self.wing_L_pts[5,:] = np.transpose(np.dot(M_now_L[3],key_pts_L3))
		M_now_R = self.state_transform_R()
		key_pts_R0 = np.transpose(self.key_pts_R)
		self.wing_R_pts[0:3,:] = np.transpose(np.dot(M_now_R[0],key_pts_R0))
		#key_pts_R1 = np.transpose(self.key_pts_R[3,:])
		#self.wing_R_pts[3,:] = np.transpose(np.dot(M_now_R[1],key_pts_R1))
		#key_pts_R2 = np.transpose(self.key_pts_R[4,:])
		#self.wing_R_pts[4,:] = np.transpose(np.dot(M_now_R[2],key_pts_R2))
		#key_pts_R3 = np.transpose(self.key_pts_R[5,:])
		#self.wing_R_pts[5,:] = np.transpose(np.dot(M_now_R[3],key_pts_R3))
		# Project 2 UV
		self.project2uv()
		for i in range(self.N_cam):
			uv_pos = np.concatenate((np.transpose(self.wing_L_uv[i]),np.transpose(self.wing_R_uv[i])),axis=0)
			self.graph_list[i].setData(pos=uv_pos, size=2, symbol=self.wing_sym, pxMode=False, text=self.wing_txt, textcolor=self.wing_clr)

	def remove_graphs(self):
		if len(self.graph_list)>0:
			for i in range(self.N_cam):
				self.v_list[i].removeItem(self.graph_list[i])
			self.graph_list = []

	def state_transform_L(self):
		q_norm = np.sqrt(pow(self.state_L[0],2)+pow(self.state_L[1],2)+pow(self.state_L[2],2)+pow(self.state_L[3],2))
		q_0 = np.array([[self.state_L[0]],
			[self.state_L[1]],
			[self.state_L[2]],
			[self.state_L[3]]])/q_norm
		T = np.array([
			[self.state_L[4]],
			[self.state_L[5]],
			[self.state_L[6]]])
		b1 = self.state_L[7]/3.0
		b2 = b1
		b3 = b1
		R_0 = self.scale_L*np.array([[2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[1],2), 2.0*q_0[1]*q_0[2]+2.0*q_0[0]*q_0[3],  2.0*q_0[1]*q_0[3]-2.0*q_0[0]*q_0[2]],
			[2.0*q_0[1]*q_0[2]-2.0*q_0[0]*q_0[3], 2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[2],2), 2.0*q_0[2]*q_0[3]+2.0*q_0[0]*q_0[1]],
			[2.0*q_0[1]*q_0[3]+2.0*q_0[0]*q_0[2], 2.0*q_0[2]*q_0[3]-2.0*q_0[0]*q_0[1], 2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[3],2)]])
		q_1 = np.array([
			[np.cos(b1/2.0)],
			[0.0],
			[np.sin(b1/2.0)],
			[0.0]])
		R_1 = np.array([[2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[1],2), 2.0*q_1[1]*q_1[2]+2.0*q_1[0]*q_1[3],  2.0*q_1[1]*q_1[3]-2.0*q_1[0]*q_1[2]],
			[2.0*q_1[1]*q_1[2]-2.0*q_1[0]*q_1[3], 2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[2],2), 2.0*q_1[2]*q_1[3]+2.0*q_1[0]*q_1[1]],
			[2.0*q_1[1]*q_1[3]+2.0*q_1[0]*q_1[2], 2.0*q_1[2]*q_1[3]-2.0*q_1[0]*q_1[1], 2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[3],2)]])
		q_2 = np.array([
			[np.cos(b2/2.0)],
			[-0.05959*np.sin(b2/2.0)],
			[0.99822*np.sin(b2/2.0)],
			[0.0]])
		R_2 = np.array([[2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[1],2), 2.0*q_2[1]*q_2[2]+2.0*q_2[0]*q_2[3],  2.0*q_2[1]*q_2[3]-2.0*q_2[0]*q_2[2]],
			[2.0*q_2[1]*q_2[2]-2.0*q_2[0]*q_2[3], 2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[2],2), 2.0*q_2[2]*q_2[3]+2.0*q_2[0]*q_2[1]],
			[2.0*q_2[1]*q_2[3]+2.0*q_2[0]*q_2[2], 2.0*q_2[2]*q_2[3]-2.0*q_2[0]*q_2[1], 2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[3],2)]])
		q_3 = np.array([
			[np.cos(b3/2.0)],
			[-0.36186*np.sin(b3/2.0)],
			[0.93223*np.sin(b3/2.0)],
			[0.0]])
		R_3 = np.array([[2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[1],2), 2.0*q_3[1]*q_3[2]+2.0*q_3[0]*q_3[3],  2.0*q_3[1]*q_3[3]-2.0*q_3[0]*q_3[2]],
			[2.0*q_3[1]*q_3[2]-2.0*q_3[0]*q_3[3], 2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[2],2), 2.0*q_3[2]*q_3[3]+2.0*q_3[0]*q_3[1]],
			[2.0*q_3[1]*q_3[3]+2.0*q_3[0]*q_3[2], 2.0*q_3[2]*q_3[3]-2.0*q_3[0]*q_3[1], 2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[3],2)]])
		# transform key_pts_0:
		M_0 = np.zeros((4,4))
		M_0[0:3,0:3] = np.squeeze(R_0)
		M_0[0:3,3] = np.squeeze(T)
		M_0[3,3] = 1.0
		# transform key_pts_1:
		M_1 = np.zeros((4,4))
		M_1[0:3,0:3] = np.dot(np.squeeze(R_1),np.squeeze(R_0))
		M_1[0:3,3] = np.squeeze(T)
		M_1[3,3] = 1.0
		# transform key_pts_2:
		M_2 = np.zeros((4,4))
		M_2[0:3,0:3] = np.dot(np.squeeze(R_2),M_1[0:3,0:3])
		M_2[0:3,3] = np.squeeze(T)
		M_2[3,3] = 1.0
		# transform key_pts_3:
		M_3 = np.zeros((4,4))
		M_3[0:3,0:3] = np.squeeze(np.dot(np.squeeze(R_3),M_2[0:3,0:3]))
		M_3[0:3,3] = np.squeeze(T)
		M_3[3,3] = 1.0
		# Return list of transformations:
		M_list = [M_0, M_1, M_2, M_3]
		return M_list

	def state_transform_R(self):
		q_norm = np.sqrt(pow(self.state_R[0],2)+pow(self.state_R[1],2)+pow(self.state_R[2],2)+pow(self.state_R[3],2))
		q_0 = np.array([[self.state_R[0]],
			[self.state_R[1]],
			[self.state_R[2]],
			[self.state_R[3]]])/q_norm
		T = np.array([
			[self.state_R[4]],
			[self.state_R[5]],
			[self.state_R[6]]])
		b1 = self.state_R[7]/3.0
		b2 = b1
		b3 = b1
		R_0 = self.scale_R*np.array([[2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[1],2), 2.0*q_0[1]*q_0[2]+2.0*q_0[0]*q_0[3],  2.0*q_0[1]*q_0[3]-2.0*q_0[0]*q_0[2]],
			[2.0*q_0[1]*q_0[2]-2.0*q_0[0]*q_0[3], 2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[2],2), 2.0*q_0[2]*q_0[3]+2.0*q_0[0]*q_0[1]],
			[2.0*q_0[1]*q_0[3]+2.0*q_0[0]*q_0[2], 2.0*q_0[2]*q_0[3]-2.0*q_0[0]*q_0[1], 2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[3],2)]])
		q_1 = np.array([
			[np.cos(b1/2.0)],
			[0.0],
			[np.sin(b1/2.0)],
			[0.0]])
		R_1 = np.array([[2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[1],2), 2.0*q_1[1]*q_1[2]+2.0*q_1[0]*q_1[3],  2.0*q_1[1]*q_1[3]-2.0*q_1[0]*q_1[2]],
			[2.0*q_1[1]*q_1[2]-2.0*q_1[0]*q_1[3], 2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[2],2), 2.0*q_1[2]*q_1[3]+2.0*q_1[0]*q_1[1]],
			[2.0*q_1[1]*q_1[3]+2.0*q_1[0]*q_1[2], 2.0*q_1[2]*q_1[3]-2.0*q_1[0]*q_1[1], 2.0*pow(q_1[0],2)-1.0+2.0*pow(q_1[3],2)]])
		q_2 = np.array([
			[np.cos(b2/2.0)],
			[0.05959*np.sin(b2/2.0)],
			[0.99822*np.sin(b2/2.0)],
			[0.0]])
		R_2 = np.array([[2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[1],2), 2.0*q_2[1]*q_2[2]+2.0*q_2[0]*q_2[3],  2.0*q_2[1]*q_2[3]-2.0*q_2[0]*q_2[2]],
			[2.0*q_2[1]*q_2[2]-2.0*q_2[0]*q_2[3], 2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[2],2), 2.0*q_2[2]*q_2[3]+2.0*q_2[0]*q_2[1]],
			[2.0*q_2[1]*q_2[3]+2.0*q_2[0]*q_2[2], 2.0*q_2[2]*q_2[3]-2.0*q_2[0]*q_2[1], 2.0*pow(q_2[0],2)-1.0+2.0*pow(q_2[3],2)]])
		q_3 = np.array([
			[np.cos(b3/2.0)],
			[0.36186*np.sin(b3/2.0)],
			[0.93223*np.sin(b3/2.0)],
			[0.0]])
		R_3 = np.array([[2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[1],2), 2.0*q_3[1]*q_3[2]+2.0*q_3[0]*q_3[3],  2.0*q_3[1]*q_3[3]-2.0*q_3[0]*q_3[2]],
			[2.0*q_3[1]*q_3[2]-2.0*q_3[0]*q_3[3], 2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[2],2), 2.0*q_3[2]*q_3[3]+2.0*q_3[0]*q_3[1]],
			[2.0*q_3[1]*q_3[3]+2.0*q_3[0]*q_3[2], 2.0*q_3[2]*q_3[3]-2.0*q_3[0]*q_3[1], 2.0*pow(q_3[0],2)-1.0+2.0*pow(q_3[3],2)]])
		# transform key_pts_0:
		M_0 = np.zeros((4,4))
		M_0[0:3,0:3] = np.squeeze(R_0)
		M_0[0:3,3] = np.squeeze(T)
		M_0[3,3] = 1.0
		# transform key_pts_1:
		M_1 = np.zeros((4,4))
		M_1[0:3,0:3] = np.dot(np.squeeze(R_1),np.squeeze(R_0))
		M_1[0:3,3] = np.squeeze(T)
		M_1[3,3] = 1.0
		# transform key_pts_2:
		M_2 = np.zeros((4,4))
		M_2[0:3,0:3] = np.dot(np.squeeze(R_2),M_1[0:3,0:3])
		M_2[0:3,3] = np.squeeze(T)
		M_2[3,3] = 1.0
		# transform key_pts_3:
		M_3 = np.zeros((4,4))
		M_3[0:3,0:3] = np.squeeze(np.dot(np.squeeze(R_3),M_2[0:3,0:3]))
		M_3[0:3,3] = np.squeeze(T)
		M_3[3,3] = 1.0
		# Return list of transformations:
		M_list = [M_0, M_1, M_2, M_3]
		return M_list

	def quat_multiply(self,qA,qB):
		QA = np.array([[qA[0],-qA[1],-qA[2],-qA[3]],
			[qA[1],qA[0],-qA[3],qA[2]],
			[qA[2],qA[3],qA[0],-qA[1]],
			[qA[3],-qA[2],qA[1],qA[0]]])
		qc = np.dot(QA,qB)
		qc = qc/np.linalg.norm(qc)
		return qc

	def update_3D_points(self,u_prev,v_prev,u_now,v_now,point_nr,cam_nr):
		xyz_prev = np.squeeze(np.dot(self.c2w_matrices[cam_nr],np.array([[u_prev+self.uv_shift[cam_nr][0,0]],[self.c_params[10,cam_nr]-v_prev-self.uv_shift[cam_nr][1,0]],[1.0]])))
		xyz_now = np.squeeze(np.dot(self.c2w_matrices[cam_nr],np.array([[u_now+self.uv_shift[cam_nr][0,0]],[self.c_params[10,cam_nr]-v_now-self.uv_shift[cam_nr][1,0]],[1.0]])))
		d_xyz = xyz_now-xyz_prev
		if point_nr == 0:
			# translation L0
			self.state_L[4] = self.state_L[4]+d_xyz[0]
			self.state_L[5] = self.state_L[5]+d_xyz[1]
			self.state_L[6] = self.state_L[6]+d_xyz[2]
		elif point_nr == 1:
			# wing stroke & deviation L2
			L2_prev = xyz_prev-np.array([self.state_L[4],self.state_L[5],self.state_L[6],1.0])
			L2_now = xyz_now-np.array([self.state_L[4],self.state_L[5],self.state_L[6],1.0])
			e_rot = np.cross(L2_now[0:3],L2_prev[0:3])/(np.linalg.norm(L2_now[0:3])*np.linalg.norm(L2_prev[0:3]))
			theta = np.arccos(np.absolute(np.dot(L2_now[0:3],L2_prev[0:3]))/(np.linalg.norm(L2_now[0:3])*np.linalg.norm(L2_prev[0:3])))
			q_update = np.array([np.cos(theta/2.0),e_rot[0]*np.sin(theta/2.0),e_rot[1]*np.sin(theta/2.0),e_rot[2]*np.sin(theta/2.0)])
			q_prev = self.state_L[0:4]
			#q_now = self.quat_multiply(q_prev,q_update)
			q_now = self.quat_multiply(q_update,q_prev)
			self.state_L[0:4] = q_now
		elif point_nr == 2:
			# translation R0
			self.state_R[4] = self.state_R[4]+d_xyz[0]
			self.state_R[5] = self.state_R[5]+d_xyz[1]
			self.state_R[6] = self.state_R[6]+d_xyz[2]
		elif point_nr == 3:
			# wing stroke & deviation R2
			R2_prev = xyz_prev-np.array([self.state_R[4],self.state_R[5],self.state_R[6],1.0])
			R2_now = xyz_now-np.array([self.state_R[4],self.state_R[5],self.state_R[6],1.0])
			e_rot = np.cross(R2_now[0:3],R2_prev[0:3])/(np.linalg.norm(R2_now[0:3])*np.linalg.norm(R2_prev[0:3]))
			theta = np.arccos(np.absolute(np.dot(R2_now[0:3],R2_prev[0:3]))/(np.linalg.norm(R2_now[0:3])*np.linalg.norm(R2_prev[0:3])))
			q_update = np.array([np.cos(theta/2.0),e_rot[0]*np.sin(theta/2.0),e_rot[1]*np.sin(theta/2.0),e_rot[2]*np.sin(theta/2.0)])
			q_prev = self.state_R[0:4]
			#q_now = self.quat_multiply(q_prev,q_update)
			q_now = self.quat_multiply(q_update,q_prev)
			self.state_R[0:4] = q_now
		self.add_wing_contours()

	def setMouseCallbacks(self):
		def onMouseDragCallback(data):
			cam_nr = int(data[3])
			point_nr = int(data[2])
			if point_nr<2:
				u_prev = self.wing_L_uv[cam_nr][0,point_nr]
				v_prev = self.wing_L_uv[cam_nr][1,point_nr]
				#print('uv_prev: ' + str(u_prev) + ', ' + str(v_prev))
				u_now = data[0]
				v_now = data[1]
				#print('uv_now: ' + str(u_now) + ', ' + str(v_now))
				self.wing_L_uv[cam_nr][0,point_nr] = u_now
				self.wing_L_uv[cam_nr][1,point_nr] = v_now
			else:
				u_prev = self.wing_R_uv[cam_nr][0,point_nr-2]
				v_prev = self.wing_R_uv[cam_nr][1,point_nr-2]
				#print('uv_prev: ' + str(u_prev) + ', ' + str(v_prev))
				u_now = data[0]
				v_now = data[1]
				#print('uv_now: ' + str(u_now) + ', ' + str(v_now))
				self.wing_R_uv[cam_nr][0,point_nr-2] = u_now
				self.wing_R_uv[cam_nr][1,point_nr-2] = v_now
			self.update_3D_points(u_prev,v_prev,u_now,v_now,point_nr,cam_nr)
			self.update_graphs()

		for i in range(self.N_cam):
			self.graph_list[i].setOnMouseDragCallback(onMouseDragCallback)

	def create_manual_track_dir(self):
		# Check if manual save directory already exists:
		try:
			os.chdir(self.output_folder)
			os.chdir(self.mov_folder)
			print("manual tracking folder exists already")
		except:
			# Create the manual tracking folder
			os.chdir(self.output_folder)
			os.mkdir(self.mov_folder)
			for j in range(self.N_cam):
				dir_name = 'cam_' + str(j+1)
				os.mkdir(dir_name)
			os.chdir(self.output_folder)
			os.chdir(self.mov_folder)
			os.mkdir('labels')
			print("created manual tracking folder")

	def save_frame(self):
		frame_list = self.load_frame(self.frame_nr)
		try:
			os.chdir(self.output_folder+'/'+self.mov_folder+'/cam_1')
		except:
			self.create_manual_track_dir()
		for i, frame in enumerate(frame_list):
			os.chdir(self.output_folder+'/'+self.mov_folder)
			os.chdir('cam_' + str(i+1))
			img_uint8 = frame*255
			img_uint8 = img_uint8.astype(np.uint8)
			cv2.imwrite('frame_' + str(self.frame_nr) + '.bmp',img_uint8)
			time.sleep(0.001)
		os.chdir(self.output_folder+'/'+self.mov_folder)
		os.chdir('labels')
		state_out = np.zeros((1,18))
		state_out[0,0] = self.scale_L
		state_out[0,1:9] = self.state_L
		state_out[0,9] = self.scale_R
		state_out[0,10:180] = self.state_R
		if path.exists('labels.h5'):
			self.hf_label_file = h5py.File('labels.h5', 'r+')
			# Check if frame already exists:
			try:
				state_dat = self.hf_label_file['label_' + str(self.frame_nr)]
				state_dat[...] = state_out
			except:
				self.hf_label_file.create_dataset('label_' + str(self.frame_nr),data=state_out)
			self.hf_label_file.close()
		else:
			self.hf_label_file = h5py.File('labels.h5', 'w')
			self.hf_label_file.create_dataset('label_' + str(self.frame_nr),data=state_out)
			self.hf_label_file.close()
		print('saved frame_' + str(self.frame_nr))

	#def save_frame(self):
	#	frame_list = self.load_frame(self.frame_nr)
	#	try:
	#		os.chdir(self.output_folder+'/'+self.mov_folder+'/cam_1')
	#	except:
	#		self.create_manual_track_dir()
	#	for i, frame in enumerate(frame_list):
	#		os.chdir(self.output_folder+'/'+self.mov_folder)
	#		os.chdir('cam_' + str(i+1))
	#		img_uint8 = frame*255
	#		img_uint8 = img_uint8.astype(np.uint8)
	#		cv2.imwrite('frame_' + str(self.frame_nr) + '.bmp',img_uint8)
	#		time.sleep(0.001)
	#	os.chdir(self.output_folder+'/'+self.mov_folder)
	#	os.chdir('labels')
	#	key_3d_mat = np.zeros((1,60))
	#	key_3d_mat[0,0:30] = np.reshape(self.wing_L_pts[:,0:3], (1,30))
	#	key_3d_mat[0,30:60] = np.reshape(self.wing_R_pts[:,0:3], (1,30))
	#	print(key_3d_mat)
	#	key_uv_mat = np.zeros((1,120))
	#	key_uv_mat[0,0:20] = np.reshape(self.wing_L_uv[0], (1,20), order='F')
	#	key_uv_mat[0,20:40] = np.reshape(self.wing_R_uv[0], (1,20), order='F')
	#	key_uv_mat[0,40:60] = np.reshape(self.wing_L_uv[1], (1,20), order='F')
	#	key_uv_mat[0,60:80] = np.reshape(self.wing_R_uv[1], (1,20), order='F')
	#	key_uv_mat[0,80:100] = np.reshape(self.wing_L_uv[1], (1,20), order='F')
	#	key_uv_mat[0,100:120] = np.reshape(self.wing_R_uv[2], (1,20), order='F')
	#	print(key_uv_mat)
	#	if path.exists('labels.h5'):
	#		self.hf_label_file = h5py.File('labels.h5', 'r+')
	#		# Check if frame already exists:
	#		try:
	#			pts_3d_dat = self.hf_label_file['key_pts_3D_' + str(self.frame_nr)]
	#			pts_3d_dat[...] = key_3d_mat
	#			pts_uv_data = self.hf_label_file['key_pts_uv_' + str(self.frame_nr)]
	#			pts_uv_dat[...] = key_uv_mat
	#		except:
	#			self.hf_label_file.create_dataset('key_pts_3D_' + str(self.frame_nr),data=key_3d_mat)
	#			self.hf_label_file.create_dataset('key_pts_uv_' + str(self.frame_nr),data=key_uv_mat)
	#		self.hf_label_file.close()
	#	else:
	#		self.hf_label_file = h5py.File('labels.h5', 'w')
	#		self.hf_label_file.create_dataset('key_pts_3D_' + str(self.frame_nr),data=key_3d_mat)
	#		self.hf_label_file.create_dataset('key_pts_uv_' + str(self.frame_nr),data=key_uv_mat)
	#		self.hf_label_file.close()
	#	print('saved frame_' + str(self.frame_nr))

	def contours2uv(self,pts_in):
		cnt_uv = []
		for n in range(self.N_cam):
			uv_pts = np.dot(self.w2c_matrices[n],pts_in)-self.uv_shift[n]
			uv_pts[1,:] = self.c_params[12,n]-uv_pts[1,:]
			cnt_uv.append(uv_pts)
		return cnt_uv

	def add_wing_contours(self):
		self.remove_wing_contours()
		# Update state
		self.state_calc.set_state(self.state_L,self.state_R)
		# Retrieve 3d coordinates left and right wings:
		wing_L_cnts = self.state_calc.wing_contour_L()
		wing_R_cnts = self.state_calc.wing_contour_R()
		# obtain 2D projections:
		cnts_L_uv = []
		for cnt in wing_L_cnts:
			cnts_L_uv.append(self.contours2uv(cnt))
		cnts_R_uv = []
		for cnt in wing_R_cnts:
			cnts_R_uv.append(self.contours2uv(cnt))
		# Add contour plots to the image items:
		self.contours_L = []
		for i,cnt_pts in enumerate(cnts_L_uv):
			for n in range(self.N_cam):
				curve_pts = np.transpose(cnt_pts[n][0:2,:])
				curve = pg.PlotCurveItem(x=curve_pts[:,0],y=curve_pts[:,1],pen=[255,0,0])
				self.contours_L.append(curve)
				self.v_list[n].addItem(self.contours_L[i*self.N_cam+n])
		self.contours_R = []
		for i,cnt_pts in enumerate(cnts_R_uv):
			for n in range(self.N_cam):
				curve_pts = np.transpose(cnt_pts[n][0:2,:])
				curve = pg.PlotCurveItem(x=curve_pts[:,0],y=curve_pts[:,1],pen=[0,0,255])
				self.contours_R.append(curve)
				self.v_list[n].addItem(self.contours_R[i*self.N_cam+n])

	def remove_wing_contours(self):
		N_L = len(self.contours_L)
		if N_L>0:
			for i in range(N_L):
				self.v_list[i%self.N_cam].removeItem(self.contours_L[i])
			self.contours_L = []
		N_R = len(self.contours_R)
		if N_R>0:
			for i in range(N_R):
				self.v_list[i%self.N_cam].removeItem(self.contours_R[i])
			self.contours_R = []

	def set_scale_L_spin(self,spin_in):
		self.scale_L_spin = spin_in
		self.scale_L_spin.setMinimum(0.1)
		self.scale_L_spin.setMaximum(2.0)
		self.scale_L_spin.setSingleStep(0.01)
		self.scale_L_spin.setValue(1.0)
		self.set_scale_L(1.0)
		self.scale_L_spin.valueChanged.connect(self.set_scale_L)

	def set_scale_L(self,scale_in):
		self.scale_L = scale_in
		self.state_calc.set_scale(self.scale_L,self.scale_R)
		self.add_wing_contours()
		self.update_graphs()
		self.display_state_L()

	def set_scale_R_spin(self,spin_in):
		self.scale_R_spin = spin_in
		self.scale_R_spin.setMinimum(0.1)
		self.scale_R_spin.setMaximum(2.0)
		self.scale_R_spin.setSingleStep(0.01)
		self.scale_R_spin.setValue(1.0)
		self.set_scale_R(1.0)
		self.scale_R_spin.valueChanged.connect(self.set_scale_R)

	def set_scale_R(self,scale_in):
		self.scale_R = scale_in
		self.state_calc.set_scale(self.scale_L,self.scale_R)
		self.add_wing_contours()
		self.update_graphs()
		self.display_state_R()

	def set_ry_L_spin(self,spin_in):
		self.ry_L_spin = spin_in
		self.ry_L_spin.setMinimum(-180.0)
		self.ry_L_spin.setMaximum(180.0)
		self.ry_L_spin.setSingleStep(1.0)
		self.ry_L_spin.setValue(0.0)
		self.set_ry_L(0.0)
		self.ry_L_spin.valueChanged.connect(self.set_ry_L)

	def set_ry_L(self,ry_in):
		# Apply rotation around wing pitch axis
		delta_ry = np.pi*((self.ry_L-ry_in)/180.0)
		self.ry_L = ry_in
		e_rot = self.wing_L_pts[1,0:3]-self.wing_L_pts[0,0:3]
		e_rot = e_rot/np.linalg.norm(e_rot)
		q_prev = self.state_L[0:4]
		q_update = np.array([np.cos(delta_ry/2.0),e_rot[0]*np.sin(delta_ry/2.0),e_rot[1]*np.sin(delta_ry/2.0),e_rot[2]*np.sin(delta_ry/2.0)])
		q_now = self.quat_multiply(q_prev,q_update)
		self.state_L[0:4] = q_now
		self.add_wing_contours()
		self.update_graphs()
		self.display_state_L()

	def set_ry_R_spin(self,spin_in):
		self.ry_R_spin = spin_in
		self.ry_R_spin.setMinimum(-180.0)
		self.ry_R_spin.setMaximum(180.0)
		self.ry_R_spin.setSingleStep(1.0)
		self.ry_R_spin.setValue(0.0)
		self.set_ry_R(0.0)
		self.ry_R_spin.valueChanged.connect(self.set_ry_R)

	def set_ry_R(self,ry_in):
		# Apply rotation around wing pitch axis
		delta_ry = np.pi*((self.ry_R-ry_in)/180.0)
		self.ry_R = ry_in
		e_rot = self.wing_R_pts[1,0:3]-self.wing_R_pts[0,0:3]
		e_rot = e_rot/np.linalg.norm(e_rot)
		q_prev = self.state_R[0:4]
		q_update = np.array([np.cos(delta_ry/2.0),e_rot[0]*np.sin(delta_ry/2.0),e_rot[1]*np.sin(delta_ry/2.0),e_rot[2]*np.sin(delta_ry/2.0)])
		q_now = self.quat_multiply(q_prev,q_update)
		self.state_R[0:4] = q_now
		self.add_wing_contours()
		self.update_graphs()
		self.display_state_R()

	def set_b1_L_spin(self,spin_in):
		self.b1_L_spin = spin_in
		self.b1_L_spin.setMinimum(-90.0)
		self.b1_L_spin.setMaximum(90.0)
		self.b1_L_spin.setSingleStep(1.0)
		self.b1_L_spin.setValue(0.0)
		self.set_b1_L(0.0)
		self.b1_L_spin.valueChanged.connect(self.set_b1_L)

	def set_b1_L(self,b1_in):
		self.state_L[7] = np.pi*(b1_in/180.0)
		self.add_wing_contours()
		self.update_graphs()
		self.display_state_L()

	def set_b1_R_spin(self,spin_in):
		self.b1_R_spin = spin_in
		self.b1_R_spin.setMinimum(-90.0)
		self.b1_R_spin.setMaximum(90.0)
		self.b1_R_spin.setSingleStep(1.0)
		self.b1_R_spin.setValue(0.0)
		self.set_b1_R(0.0)
		self.b1_R_spin.valueChanged.connect(self.set_b1_R)

	def set_b1_R(self,b1_in):
		self.state_R[7] = np.pi*(b1_in/180.0)
		self.add_wing_contours()
		self.update_graphs()
		self.display_state_R()

	def reset_L(self):
		self.scale_L_spin.setValue(1.0)
		self.set_scale_L(1.0)
		self.ry_L_spin.setValue(0.0)
		self.set_ry_L(0.0)
		self.b1_L_spin.setValue(0.0)
		self.set_b1_L(0.0)
		self.state_L = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.add_wing_contours()
		self.update_graphs()
		self.display_state_L()

	def reset_R(self):
		self.scale_R_spin.setValue(1.0)
		self.set_scale_R(1.0)
		self.ry_R_spin.setValue(0.0)
		self.set_ry_R(0.0)
		self.b1_R_spin.setValue(0.0)
		self.set_b1_R(0.0)
		self.state_R = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.add_wing_contours()
		self.update_graphs()
		self.display_state_R()

	def set_display_state_L(self,table_in):
		self.state_L_table = table_in
		self.state_L_table.setRowCount(9)
		self.state_L_table.setColumnCount(2)
		self.state_L_table.setItem(0,0,QTableWidgetItem('Scale:'))
		self.state_L_table.setItem(0,1,QTableWidgetItem(str(self.scale_L)))
		self.state_L_table.setItem(1,0,QTableWidgetItem('q0:'))
		self.state_L_table.setItem(1,1,QTableWidgetItem(str(self.state_L[0])))
		self.state_L_table.setItem(2,0,QTableWidgetItem('q1:'))
		self.state_L_table.setItem(2,1,QTableWidgetItem(str(self.state_L[1])))
		self.state_L_table.setItem(3,0,QTableWidgetItem('q2:'))
		self.state_L_table.setItem(3,1,QTableWidgetItem(str(self.state_L[2])))
		self.state_L_table.setItem(4,0,QTableWidgetItem('q3:'))
		self.state_L_table.setItem(4,1,QTableWidgetItem(str(self.state_L[3])))
		self.state_L_table.setItem(5,0,QTableWidgetItem('tx:'))
		self.state_L_table.setItem(5,1,QTableWidgetItem(str(self.state_L[4])))
		self.state_L_table.setItem(6,0,QTableWidgetItem('ty:'))
		self.state_L_table.setItem(6,1,QTableWidgetItem(str(self.state_L[5])))
		self.state_L_table.setItem(7,0,QTableWidgetItem('tz:'))
		self.state_L_table.setItem(7,1,QTableWidgetItem(str(self.state_L[6])))
		self.state_L_table.setItem(8,0,QTableWidgetItem('beta:'))
		self.state_L_table.setItem(8,1,QTableWidgetItem(str(self.state_L[7])))
		self.state_L_table.resizeColumnsToContents()

	def display_state_L(self):
		self.state_L_table.setItem(0,0,QTableWidgetItem('Scale:'))
		self.state_L_table.setItem(0,1,QTableWidgetItem(str(self.scale_L)))
		self.state_L_table.setItem(1,0,QTableWidgetItem('q0:'))
		self.state_L_table.setItem(1,1,QTableWidgetItem(str(self.state_L[0])))
		self.state_L_table.setItem(2,0,QTableWidgetItem('q1:'))
		self.state_L_table.setItem(2,1,QTableWidgetItem(str(self.state_L[1])))
		self.state_L_table.setItem(3,0,QTableWidgetItem('q2:'))
		self.state_L_table.setItem(3,1,QTableWidgetItem(str(self.state_L[2])))
		self.state_L_table.setItem(4,0,QTableWidgetItem('q3:'))
		self.state_L_table.setItem(4,1,QTableWidgetItem(str(self.state_L[3])))
		self.state_L_table.setItem(5,0,QTableWidgetItem('tx:'))
		self.state_L_table.setItem(5,1,QTableWidgetItem(str(self.state_L[4])))
		self.state_L_table.setItem(6,0,QTableWidgetItem('ty:'))
		self.state_L_table.setItem(6,1,QTableWidgetItem(str(self.state_L[5])))
		self.state_L_table.setItem(7,0,QTableWidgetItem('tz:'))
		self.state_L_table.setItem(7,1,QTableWidgetItem(str(self.state_L[6])))
		self.state_L_table.setItem(8,0,QTableWidgetItem('beta:'))
		self.state_L_table.setItem(8,1,QTableWidgetItem(str(self.state_L[7])))
		self.state_L_table.resizeColumnsToContents()

	def set_display_state_R(self,table_in):
		self.state_R_table = table_in
		self.state_R_table.setRowCount(9)
		self.state_R_table.setColumnCount(2)
		self.state_R_table.setItem(0,0,QTableWidgetItem('Scale:'))
		self.state_R_table.setItem(0,1,QTableWidgetItem(str(self.scale_L)))
		self.state_R_table.setItem(1,0,QTableWidgetItem('q0:'))
		self.state_R_table.setItem(1,1,QTableWidgetItem(str(self.state_L[0])))
		self.state_R_table.setItem(2,0,QTableWidgetItem('q1:'))
		self.state_R_table.setItem(2,1,QTableWidgetItem(str(self.state_L[1])))
		self.state_R_table.setItem(3,0,QTableWidgetItem('q2:'))
		self.state_R_table.setItem(3,1,QTableWidgetItem(str(self.state_L[2])))
		self.state_R_table.setItem(4,0,QTableWidgetItem('q3:'))
		self.state_R_table.setItem(4,1,QTableWidgetItem(str(self.state_L[3])))
		self.state_R_table.setItem(5,0,QTableWidgetItem('tx:'))
		self.state_R_table.setItem(5,1,QTableWidgetItem(str(self.state_L[4])))
		self.state_R_table.setItem(6,0,QTableWidgetItem('ty:'))
		self.state_R_table.setItem(6,1,QTableWidgetItem(str(self.state_L[5])))
		self.state_R_table.setItem(7,0,QTableWidgetItem('tz:'))
		self.state_R_table.setItem(7,1,QTableWidgetItem(str(self.state_L[6])))
		self.state_R_table.setItem(8,0,QTableWidgetItem('beta:'))
		self.state_R_table.setItem(8,1,QTableWidgetItem(str(self.state_L[7])))
		self.state_R_table.resizeColumnsToContents()

	def display_state_R(self):
		self.state_R_table.setItem(0,0,QTableWidgetItem('Scale:'))
		self.state_R_table.setItem(0,1,QTableWidgetItem(str(self.scale_R)))
		self.state_R_table.setItem(1,0,QTableWidgetItem('q0:'))
		self.state_R_table.setItem(1,1,QTableWidgetItem(str(self.state_R[0])))
		self.state_R_table.setItem(2,0,QTableWidgetItem('q1:'))
		self.state_R_table.setItem(2,1,QTableWidgetItem(str(self.state_R[1])))
		self.state_R_table.setItem(3,0,QTableWidgetItem('q2:'))
		self.state_R_table.setItem(3,1,QTableWidgetItem(str(self.state_R[2])))
		self.state_R_table.setItem(4,0,QTableWidgetItem('q3:'))
		self.state_R_table.setItem(4,1,QTableWidgetItem(str(self.state_R[3])))
		self.state_R_table.setItem(5,0,QTableWidgetItem('tx:'))
		self.state_R_table.setItem(5,1,QTableWidgetItem(str(self.state_R[4])))
		self.state_R_table.setItem(6,0,QTableWidgetItem('ty:'))
		self.state_R_table.setItem(6,1,QTableWidgetItem(str(self.state_R[5])))
		self.state_R_table.setItem(7,0,QTableWidgetItem('tz:'))
		self.state_R_table.setItem(7,1,QTableWidgetItem(str(self.state_R[6])))
		self.state_R_table.setItem(8,0,QTableWidgetItem('beta:'))
		self.state_R_table.setItem(8,1,QTableWidgetItem(str(self.state_R[7])))
		self.state_R_table.resizeColumnsToContents()