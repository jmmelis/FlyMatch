from __future__ import print_function
import sys
#import vtk
from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTreeView, QFileSystemModel, QTableWidget, QTableWidgetItem, QVBoxLayout, QFileDialog
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
#import pickle
import os
import os.path
import math
import copy
import time

#from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from session_select import Ui_Session_Dialog
from fly_match_ui import Ui_MainWindow

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

# Main GUI class:

class FlyMatch(QtWidgets.QMainWindow, Ui_MainWindow, QObject):

	def __init__(self, parent=None):
		super(FlyMatch,self).__init__(parent)
		self.setupUi(self)

		self.N_mov = 1
		self.N_cam = 3
		self.start_frame = 0
		self.end_frame = 16375
		self.session_path = ''
		self.session_folder = ''
		self.mov_folder = ''
		self.output_path = ''
		self.output_folder = ''

		self.select_seq_window = SelectFolderWindow('/media/flyami/flyami_hdd_1')
		self.select_seq_window.setWindowTitle("Select sequence folder")
		self.select_mov_window = SelectFolderWindow('/media/flyami/flyami_hdd_1')
		self.select_mov_window.setWindowTitle("Select movie folder")

		self.select_movie()
		self.select_frame()

	def select_movie(self):
		# Select session folder
		self.seq_select_btn.clicked.connect(self.select_seq_callback)
		# Select movie folder
		self.mov_select_btn.clicked.connect(self.select_mov_callback)

	def select_frame(self):
		self.load_frames_btn.clicked.connect(self.load_frames)

	def load_frames(self):
		self.model_fit()

	def select_seq_callback(self):
		self.select_seq_window.exec_()
		self.session_path = str(self.select_seq_window.folder_path)
		print(self.session_path)
		self.session_folder = str(self.select_seq_window.folder_name)
		self.seq_disp.setText(self.session_folder)
		self.select_mov_window.update_file_model(self.session_path)
		#self.select_out_window.update_file_model(self.session_path)

	def select_mov_callback(self):
		self.select_mov_window.exec_()
		self.mov_folder = str(self.select_mov_window.folder_name)
		print(self.mov_folder)
		self.mov_disp.setText(self.mov_folder)

	def model_fit(self):
		self.frame_viewer.set_session_folder(self.session_path)
		self.frame_viewer.set_mov_folder(self.mov_folder)
		# Set calibration folder:
		self.cal_file = self.session_path + '/calibration/cam_calib.txt'
		self.frame_viewer.set_calibration_loc(self.cal_file,0.040,0.5,175.0)
		# Create manual tracking directory:
		#self.frame_viewer.create_manual_track_dir()
		# Load model:
		#self.frame_viewer.load_model(1)
		# Frame spin:
		self.frame_spin.setMinimum(self.start_frame)
		self.frame_spin.setMaximum(self.end_frame)
		self.frame_spin.setValue(self.start_frame)
		self.frame_viewer.add_frame(self.start_frame)
		self.frame_spin.valueChanged.connect(self.frame_viewer.update_frame)
		# Add state displays:
		self.frame_viewer.set_display_state_L(self.state_L_table)
		self.frame_viewer.set_display_state_R(self.state_R_table)
		# Add graphs
		self.frame_viewer.add_graphs()
		self.frame_viewer.setMouseCallbacks()
		# uv_correction:
		self.frame_viewer.set_u_cam_1_spin(self.u_cam1_spin)
		self.frame_viewer.set_v_cam_1_spin(self.v_cam1_spin)
		self.frame_viewer.set_u_cam_2_spin(self.u_cam2_spin)
		self.frame_viewer.set_v_cam_2_spin(self.v_cam2_spin)
		self.frame_viewer.set_u_cam_3_spin(self.u_cam3_spin)
		self.frame_viewer.set_v_cam_3_spin(self.v_cam3_spin)
		self.save_uv_shift_btn.clicked.connect(self.frame_viewer.save_uv_shift)
		#
		self.frame_viewer.set_scale_L_spin(self.L_scale_spin)
		self.frame_viewer.set_ry_L_spin(self.ry_L_spin)
		self.frame_viewer.set_b1_L_spin(self.b1_L_spin)
		self.reset_L_btn.clicked.connect(self.frame_viewer.reset_L)
		#
		self.frame_viewer.set_scale_R_spin(self.R_scale_spin)
		self.frame_viewer.set_ry_R_spin(self.ry_R_spin)
		self.frame_viewer.set_b1_R_spin(self.b1_R_spin)
		self.reset_R_btn.clicked.connect(self.frame_viewer.reset_R)
		# Save label:
		self.save_lbl_btn.clicked.connect(self.frame_viewer.save_frame)

# -------------------------------------------------------------------------------------------------

def appMain():
	app = QtWidgets.QApplication(sys.argv)
	mainWindow = FlyMatch()
	mainWindow.show()
	app.exec_()

# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	appMain()