import numpy as np
import numpy.matlib
from scipy.optimize import least_squares
from sklearn.decomposition import PCA

class StateFitter():

	def __init__(self):
		# Initial state:
		#self.init_state_L = [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
		#self.init_state_R = [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
		#self.init_state_L = [1.0,0.0,0.0,0.0,0.0,0.0,0.0]
		#self.init_state_R = [1.0,0.0,0.0,0.0,0.0,0.0,0.0]
		self.state_L = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		self.state_R = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		#self.lower_bnds = [-1.0,-1.0,-1.0,-1.0,-10.0,-10.0,-10.0,-0.2*np.pi,-0.3*np.pi,-0.4*np.pi]
		#self.upper_bnds = [1.0,1.0,1.0,1.0,10.0,10.0,10.0,0.2*np.pi,0.3*np.pi,0.4*np.pi]
		#self.lower_bnds = [-1.0,-1.0,-1.0,-1.0,-10.0,-10.0,-10.0]
		#self.upper_bnds = [1.0,1.0,1.0,1.0,10.0,10.0,10.0]
		#self.weights = np.array([1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,0.5,0.5,0.5,
		#	0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1.0])
		self.scale_L = 1.0
		self.scale_R = 1.0

	def wing_contour_L(self):
		state_in = self.state_L
		scl = self.scale_L
		L0_vein = np.transpose(np.array([[scl*0.0578,scl*0.0145,0.0,1.0],
			[scl*0.1301,scl*0.0940,0.0,1.0],
			[scl*0.1735,scl*0.1663,0.0,1.0],
			[scl*0.2024,scl*0.2386,0.0,1.0],
			[scl*0.2241,scl*0.3108,0.0,1.0],
			[scl*0.2313,scl*0.3831,0.0,1.0],
			[scl*0.2458,scl*0.4554,0.0,1.0],
			[scl*0.2458,scl*0.5277,0.0,1.0],
			[scl*0.2313,scl*0.5711,0.0,1.0]]))
		L1_vein = np.transpose(np.array([[scl*0.0578,scl*0.0145,0.0,1.0],
			[scl*0.0361,scl*0.0940,0.0,1.0],
			[scl*0.0361,scl*0.1663,0.0,1.0],
			[scl*0.0506,scl*0.2386,0.0,1.0],
			[scl*0.0651,scl*0.3109,0.0,1.0],
			[scl*0.1012,scl*0.3831,0.0,1.0],
			[scl*0.1374,scl*0.4554,0.0,1.0],
			[scl*0.1952,scl*0.5277,0.0,1.0],
			[scl*0.2386,scl*0.5639,0.0,1.0],
			[scl*0.2458,scl*0.6000,0.0,1.0],
			[scl*0.2747,scl*0.6723,0.0,1.0],
			[scl*0.3036,scl*0.7446,0.0,1.0],
			[scl*0.3181,scl*0.8169,0.0,1.0],
			[scl*0.3398,scl*0.8892,0.0,1.0],
			[scl*0.3542,scl*0.9615,0.0,1.0],
			[scl*0.3687,scl*1.0337,0.0,1.0],
			[scl*0.3831,scl*1.1060,0.0,1.0],
			[scl*0.3904,scl*1.1783,0.0,1.0],
			[scl*0.3976,scl*1.2506,0.0,1.0],
			[scl*0.4048,scl*1.3229,0.0,1.0],
			[scl*0.4121,scl*1.3952,0.0,1.0],
			[scl*0.4193,scl*1.4675,0.0,1.0],
			[scl*0.4265,scl*1.6121,0.0,1.0],
			[scl*0.4265,scl*1.6844,0.0,1.0],
			[scl*0.4337,scl*1.7566,0.0,1.0],
			[scl*0.4265,scl*1.8289,0.0,1.0],
			[scl*0.4193,scl*1.9012,0.0,1.0],
			[scl*0.4121,scl*1.9735,0.0,1.0],
			[scl*0.4048,scl*2.0458,0.0,1.0],
			[scl*0.3904,scl*2.1181,0.0,1.0],
			[scl*0.3759,scl*2.1904,0.0,1.0],
			[scl*0.3542,scl*2.2627,0.0,1.0],
			[scl*0.3253,scl*2.3350,0.0,1.0],
			[scl*0.2819,scl*2.4073,0.0,1.0],
			[scl*0.2241,scl*2.4795,0.0,1.0],
			[scl*0.1374,scl*2.5518,0.0,1.0],
			[scl*0.0867,scl*2.5880,0.0,1.0],
			[scl*0.0506,scl*2.6097,0.0,1.0],
			[scl*0.0,scl*2.6241,0.0,1.0]]))
		L2_vein = np.transpose(np.array([[scl*0.0361,scl*0.0940,0.0,1.0],
			[scl*-0.0145,scl*0.1663,0.0,1.0],
			[scl*-0.0145,scl*0.2241,0.0,1.0],
			[scl*0.0145,scl*0.3109,0.0,1.0],
			[scl*0.0506,scl*0.3831,0.0,1.0],
			[scl*0.0723,scl*0.4554,0.0,1.0],
			[scl*0.0795,scl*0.5277,0.0,1.0],
			[scl*0.0867,scl*0.6000,0.0,1.0],
			[scl*0.1084,scl*0.6723,0.0,1.0],
			[scl*0.1229,scl*0.7446,0.0,1.0],
			[scl*0.1374,scl*0.8169,0.0,1.0],
			[scl*0.1518,scl*0.8892,0.0,1.0],
			[scl*0.1735,scl*0.9615,0.0,1.0],
			[scl*0.1880,scl*1.0337,0.0,1.0],
			[scl*0.2096,scl*1.1060,0.0,1.0],
			[scl*0.2241,scl*1.1783,0.0,1.0],
			[scl*0.2386,scl*1.2506,0.0,1.0],
			[scl*0.2458,scl*1.3229,0.0,1.0],
			[scl*0.2602,scl*1.3952,0.0,1.0],
			[scl*0.2675,scl*1.4675,0.0,1.0],
			[scl*0.2747,scl*1.5398,0.0,1.0],
			[scl*0.2747,scl*1.6121,0.0,1.0],
			[scl*0.2747,scl*1.6844,0.0,1.0],
			[scl*0.2747,scl*1.7566,0.0,1.0],
			[scl*0.2819,scl*1.8289,0.0,1.0],
			[scl*0.2819,scl*1.9012,0.0,1.0],
			[scl*0.2819,scl*1.9735,0.0,1.0],
			[scl*0.2819,scl*2.0458,0.0,1.0],
			[scl*0.2892,scl*2.1181,0.0,1.0],
			[scl*0.2892,scl*2.1904,0.0,1.0],
			[scl*0.3036,scl*2.2627,0.0,1.0],
			[scl*0.3253,scl*2.3205,0.0,1.0]]))
		L3_vein = np.transpose(np.array([
			[scl*-0.0145,scl*0.2241,0.0,1.0],
			[scl*0.0,scl*0.3108,0.0,1.0],
			[scl*0.0145,scl*0.3831,0.0,1.0],
			[scl*0.0145,scl*0.4554,0.0,1.0],
			[scl*0.0072,scl*0.5277,0.0,1.0],
			[scl*0.0072,scl*0.6000,0.0,1.0],
			[scl*0.0,scl*0.6723,0.0,1.0],
			[scl*0.0072,scl*0.7157,0.0,1.0],
			[scl*0.0145,scl*0.7446,0.0,1.0],
			[scl*0.0217,scl*0.8169,0.0,1.0],
			[scl*0.0361,scl*0.8892,0.0,1.0],
			[scl*0.0434,scl*0.9615,0.0,1.0],
			[scl*0.0506,scl*1.0337,0.0,1.0],
			[scl*0.0578,scl*1.1060,0.0,1.0],
			[scl*0.0578,scl*1.1783,0.0,1.0],
			[scl*0.0578,scl*1.2506,0.0,1.0],
			[scl*0.0578,scl*1.3229,0.0,1.0],
			[scl*0.0578,scl*1.3952,0.0,1.0],
			[scl*0.0651,scl*1.4675,0.0,1.0],
			[scl*0.0651,scl*1.5398,0.0,1.0],
			[scl*0.0723,scl*1.6121,0.0,1.0],
			[scl*0.0723,scl*1.6844,0.0,1.0],
			[scl*0.0723,scl*1.7566,0.0,1.0],
			[scl*0.0723,scl*1.8289,0.0,1.0],
			[scl*0.0651,scl*1.9012,0.0,1.0],
			[scl*0.0651,scl*1.9735,0.0,1.0],
			[scl*0.0578,scl*2.0458,0.0,1.0],
			[scl*0.0578,scl*2.1181,0.0,1.0],
			[scl*0.0506,scl*2.1904,0.0,1.0],
			[scl*0.0506,scl*2.2627,0.0,1.0],
			[scl*0.0361,scl*2.3350,0.0,1.0],
			[scl*0.0217,scl*2.4073,0.0,1.0],
			[scl*0.0145,scl*2.4795,0.0,1.0],
			[scl*0.0072,scl*2.5518,0.0,1.0],
			[scl*0.0,scl*2.6241,0.0,1.0]]))
		L4_vein = np.transpose(np.array([
			[scl*-0.0867,scl*0.0145,0.0,1.0],
			[scl*-0.0651,scl*0.0940,0.0,1.0],
			[scl*-0.0506,scl*0.1663,0.0,1.0],
			[scl*-0.0723,scl*0.2386,0.0,1.0],
			[scl*-0.0795,scl*0.3108,0.0,1.0],
			[scl*-0.0795,scl*0.3831,0.0,1.0],
			[scl*-0.0867,scl*0.4554,0.0,1.0],
			[scl*-0.0867,scl*0.5277,0.0,1.0],
			[scl*-0.0940,scl*0.6000,0.0,1.0],
			[scl*-0.0867,scl*0.6723,0.0,1.0],
			[scl*-0.0795,scl*0.6940,0.0,1.0],
			[scl*-0.1012,scl*0.7446,0.0,1.0],
			[scl*-0.1084,scl*0.8169,0.0,1.0],
			[scl*-0.1301,scl*0.8892,0.0,1.0],
			[scl*-0.1446,scl*0.9615,0.0,1.0],
			[scl*-0.1518,scl*1.0337,0.0,1.0],
			[scl*-0.1663,scl*1.1060,0.0,1.0],
			[scl*-0.1807,scl*1.1783,0.0,1.0],
			[scl*-0.1880,scl*1.2506,0.0,1.0],
			[scl*-0.2024,scl*1.2868,0.0,1.0],
			[scl*-0.1952,scl*1.3229,0.0,1.0],
			[scl*-0.1880,scl*1.3952,0.0,1.0],
			[scl*-0.1880,scl*1.4675,0.0,1.0],
			[scl*-0.1880,scl*1.5398,0.0,1.0],
			[scl*-0.1880,scl*1.6121,0.0,1.0],
			[scl*-0.1880,scl*1.6844,0.0,1.0],
			[scl*-0.1880,scl*1.7566,0.0,1.0],
			[scl*-0.1880,scl*1.8289,0.0,1.0],
			[scl*-0.1880,scl*1.9012,0.0,1.0],
			[scl*-0.1952,scl*1.9735,0.0,1.0],
			[scl*-0.1952,scl*2.0458,0.0,1.0],
			[scl*-0.1952,scl*2.1181,0.0,1.0],
			[scl*-0.1952,scl*2.1904,0.0,1.0],
			[scl*-0.2024,scl*2.2627,0.0,1.0],
			[scl*-0.2024,scl*2.3350,0.0,1.0],
			[scl*-0.2096,scl*2.4073,0.0,1.0],
			[scl*-0.2169,scl*2.4795,0.0,1.0],
			[scl*-0.2386,scl*2.5591,0.0,1.0]]))
		L5_vein = np.transpose(np.array([
			[scl*-0.0867,scl*0.0145,0.0,1.0],
			[scl*-0.1229,scl*0.0940,0.0,1.0],
			[scl*-0.1446,scl*0.1663,0.0,1.0],
			[scl*-0.1663,scl*0.2386,0.0,1.0],
			[scl*-0.1807,scl*0.3108,0.0,1.0],
			[scl*-0.1952,scl*0.3831,0.0,1.0],
			[scl*-0.2024,scl*0.4554,0.0,1.0],
			[scl*-0.2241,scl*0.5277,0.0,1.0],
			[scl*-0.2386,scl*0.6000,0.0,1.0],
			[scl*-0.2602,scl*0.6723,0.0,1.0],
			[scl*-0.2747,scl*0.7446,0.0,1.0],
			[scl*-0.2892,scl*0.8169,0.0,1.0],
			[scl*-0.3108,scl*0.8892,0.0,1.0],
			[scl*-0.3253,scl*0.9615,0.0,1.0],
			[scl*-0.3470,scl*1.0337,0.0,1.0],
			[scl*-0.3615,scl*1.1060,0.0,1.0],
			[scl*-0.3904,scl*1.1783,0.0,1.0],
			[scl*-0.4048,scl*1.2506,0.0,1.0],
			[scl*-0.4410,scl*1.3229,0.0,1.0],
			[scl*-0.4988,scl*1.4313,0.0,1.0],
			[scl*-0.5494,scl*1.4675,0.0,1.0],
			[scl*-0.6217,scl*1.5398,0.0,1.0],
			[scl*-0.7012,scl*1.5976,0.0,1.0]]))
		C1_vein = np.transpose(np.array([
			[scl*0.0,scl*2.6241,0.0,1.0],
			[scl*-0.0578,scl*2.6241,0.0,1.0],
			[scl*-0.1374,scl*2.6097,0.0,1.0],
			[scl*-0.1880,scl*2.5880,0.0,1.0],
			[scl*-0.2386,scl*2.5518,0.0,1.0]]))
		C2_vein = np.transpose(np.array([
			[scl*-0.2386,scl*2.5518,0.0,1.0],
			[scl*-0.3181,scl*2.4795,0.0,1.0],
			[scl*-0.3904,scl*2.4073,0.0,1.0],
			[scl*-0.4410,scl*2.3350,0.0,1.0],
			[scl*-0.4916,scl*2.2627,0.0,1.0],
			[scl*-0.5277,scl*2.1904,0.0,1.0],
			[scl*-0.5566,scl*2.1181,0.0,1.0],
			[scl*-0.5928,scl*2.0458,0.0,1.0],
			[scl*-0.6145,scl*1.9735,0.0,1.0],
			[scl*-0.6362,scl*1.9012,0.0,1.0],
			[scl*-0.6578,scl*1.8289,0.0,1.0],
			[scl*-0.6795,scl*1.7566,0.0,1.0],
			[scl*-0.7012,scl*1.6844,0.0,1.0],
			[scl*-0.7012,scl*1.6121,0.0,1.0],
			[scl*-0.7012,scl*1.5976,0.0,1.0]]))
		C3_vein = np.transpose(np.array([
			[scl*-0.7012,scl*1.5976,0.0,1.0],
			[scl*-0.7229,scl*1.5398,0.0,1.0],
			[scl*-0.7446,scl*1.4675,0.0,1.0],
			[scl*-0.7518,scl*1.3952,0.0,1.0],
			[scl*-0.7663,scl*1.3229,0.0,1.0],
			[scl*-0.7735,scl*1.2506,0.0,1.0],
			[scl*-0.7807,scl*1.1783,0.0,1.0],
			[scl*-0.7807,scl*1.1060,0.0,1.0],
			[scl*-0.7807,scl*1.0337,0.0,1.0],
			[scl*-0.7880,scl*0.9615,0.0,1.0],
			[scl*-0.7880,scl*0.8892,0.0,1.0],
			[scl*-0.7880,scl*0.8169,0.0,1.0],
			[scl*-0.7807,scl*0.7446,0.0,1.0],
			[scl*-0.7663,scl*0.6723,0.0,1.0],
			[scl*-0.7590,scl*0.6000,0.0,1.0],
			[scl*-0.7446,scl*0.5277,0.0,1.0],
			[scl*-0.7229,scl*0.4554,0.0,1.0],
			[scl*-0.6940,scl*0.3831,0.0,1.0],
			[scl*-0.6434,scl*0.3108,0.0,1.0],
			[scl*-0.5566,scl*0.2386,0.0,1.0],
			[scl*-0.3831,scl*0.1663,0.0,1.0],
			[scl*-0.2169,scl*0.0940,0.0,1.0],
			[scl*-0.1157,scl*0.0217,0.0,1.0],
			[scl*-0.0867,scl*0.0145,0.0,1.0]]))
		A_vein = np.transpose(np.array([[scl*0.0072,scl*0.7157,0.0,1.0],
			[scl*-0.0795,scl*0.6940,0.0,1.0]]))
		P_vein = np.transpose(np.array([[scl*-0.1952,scl*1.2868,0.0,1.0],
			[scl*-0.3325,scl*1.2868,0.0,1.0],
			[scl*-0.4048,scl*1.2578,0.0,1.0]]))
		q_norm = np.sqrt(pow(state_in[0],2)+pow(state_in[1],2)+pow(state_in[2],2)+pow(state_in[3],2))
		q_0 = np.array([[state_in[0]/q_norm],
			[state_in[1]/q_norm],
			[state_in[2]/q_norm],
			[state_in[3]/q_norm]])
		T = np.array([
			[state_in[4]],
			[state_in[5]],
			[state_in[6]]])
		b1 = state_in[7]/3.0
		b2 = b1
		b3 = b1
		R_0 = np.array([[2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[1],2), 2.0*q_0[1]*q_0[2]+2.0*q_0[0]*q_0[3],  2.0*q_0[1]*q_0[3]-2.0*q_0[0]*q_0[2]],
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
		# transform L0, L1, L2:
		M_0 = np.zeros((4,4))
		M_0[0:3,0:3] = np.squeeze(R_0)
		M_0[0:3,3] = np.squeeze(T)
		M_0[3,3] = 1.0
		L0_pts = np.dot(M_0,L0_vein)
		L1_pts = np.dot(M_0,L1_vein)
		L2_pts = np.dot(M_0,L2_vein)
		# transform L3 ,A and C1:
		M_1 = np.zeros((4,4))
		#M_1[0:3,0:3] = np.dot(np.squeeze(R_1),np.squeeze(R_0))
		M_1[0:3,0:3] = np.dot(np.squeeze(R_0),np.squeeze(R_1))
		M_1[0:3,3] = np.squeeze(T)
		M_1[3,3] = 1.0
		L3_pts = np.dot(M_1,L3_vein)
		A_pts = np.dot(M_1,A_vein)
		C1_pts = np.dot(M_1,C1_vein)
		# transform L4, C2 and P:
		M_2 = np.zeros((4,4))
		#M_2[0:3,0:3] = np.dot(np.squeeze(R_2),M_1[0:3,0:3])
		M_2[0:3,0:3] = np.dot(M_1[0:3,0:3],np.squeeze(R_2))
		M_2[0:3,3] = np.squeeze(T)
		M_2[3,3] = 1.0
		L4_pts = np.dot(M_2,L4_vein)
		C2_pts = np.dot(M_2,C2_vein)
		P_pts = np.dot(M_2,P_vein)
		# transform L5 and C3:
		M_3 = np.zeros((4,4))
		#M_3[0:3,0:3] = np.squeeze(np.dot(np.squeeze(R_3),M_2[0:3,0:3]))
		M_3[0:3,0:3] = np.squeeze(np.dot(M_2[0:3,0:3],np.squeeze(R_3)))
		M_3[0:3,3] = np.squeeze(T)
		M_3[3,3] = 1.0
		L5_pts = np.dot(M_3,L5_vein)
		C3_pts = np.dot(M_3,C3_vein)
		contour_list = [L0_pts,L1_pts,L2_pts,L3_pts,L4_pts,L5_pts,C1_pts,C2_pts,C3_pts,A_pts,P_pts]
		return contour_list

	def wing_contour_R(self):
		state_in = self.state_R
		scl = self.scale_R
		L0_vein = np.transpose(np.array([[scl*0.0578,scl*-0.0145,0.0,1.0],
			[scl*0.1301,scl*-0.0940,0.0,1.0],
			[scl*0.1735,scl*-0.1663,0.0,1.0],
			[scl*0.2024,scl*-0.2386,0.0,1.0],
			[scl*0.2241,scl*-0.3108,0.0,1.0],
			[scl*0.2313,scl*-0.3831,0.0,1.0],
			[scl*0.2458,scl*-0.4554,0.0,1.0],
			[scl*0.2458,scl*-0.5277,0.0,1.0],
			[scl*0.2313,scl*-0.5711,0.0,1.0]]))
		L1_vein = np.transpose(np.array([[scl*0.0578,scl*-0.0145,0.0,1.0],
			[scl*0.0361,scl*-0.0940,0.0,1.0],
			[scl*0.0361,scl*-0.1663,0.0,1.0],
			[scl*0.0506,scl*-0.2386,0.0,1.0],
			[scl*0.0651,scl*-0.3109,0.0,1.0],
			[scl*0.1012,scl*-0.3831,0.0,1.0],
			[scl*0.1374,scl*-0.4554,0.0,1.0],
			[scl*0.1952,scl*-0.5277,0.0,1.0],
			[scl*0.2386,scl*-0.5639,0.0,1.0],
			[scl*0.2458,scl*-0.6000,0.0,1.0],
			[scl*0.2747,scl*-0.6723,0.0,1.0],
			[scl*0.3036,scl*-0.7446,0.0,1.0],
			[scl*0.3181,scl*-0.8169,0.0,1.0],
			[scl*0.3398,scl*-0.8892,0.0,1.0],
			[scl*0.3542,scl*-0.9615,0.0,1.0],
			[scl*0.3687,scl*-1.0337,0.0,1.0],
			[scl*0.3831,scl*-1.1060,0.0,1.0],
			[scl*0.3904,scl*-1.1783,0.0,1.0],
			[scl*0.3976,scl*-1.2506,0.0,1.0],
			[scl*0.4048,scl*-1.3229,0.0,1.0],
			[scl*0.4121,scl*-1.3952,0.0,1.0],
			[scl*0.4193,scl*-1.4675,0.0,1.0],
			[scl*0.4265,scl*-1.6121,0.0,1.0],
			[scl*0.4265,scl*-1.6844,0.0,1.0],
			[scl*0.4337,scl*-1.7566,0.0,1.0],
			[scl*0.4265,scl*-1.8289,0.0,1.0],
			[scl*0.4193,scl*-1.9012,0.0,1.0],
			[scl*0.4121,scl*-1.9735,0.0,1.0],
			[scl*0.4048,scl*-2.0458,0.0,1.0],
			[scl*0.3904,scl*-2.1181,0.0,1.0],
			[scl*0.3759,scl*-2.1904,0.0,1.0],
			[scl*0.3542,scl*-2.2627,0.0,1.0],
			[scl*0.3253,scl*-2.3350,0.0,1.0],
			[scl*0.2819,scl*-2.4073,0.0,1.0],
			[scl*0.2241,scl*-2.4795,0.0,1.0],
			[scl*0.1374,scl*-2.5518,0.0,1.0],
			[scl*0.0867,scl*-2.5880,0.0,1.0],
			[scl*0.0506,scl*-2.6097,0.0,1.0],
			[scl*0.0,scl*-2.6241,0.0,1.0]]))
		L2_vein = np.transpose(np.array([[scl*0.0361,scl*-0.0940,0.0,1.0],
			[scl*-0.0145,scl*-0.1663,0.0,1.0],
			[scl*-0.0145,scl*-0.2241,0.0,1.0],
			[scl*0.0145,scl*-0.3109,0.0,1.0],
			[scl*0.0506,scl*-0.3831,0.0,1.0],
			[scl*0.0723,scl*-0.4554,0.0,1.0],
			[scl*0.0795,scl*-0.5277,0.0,1.0],
			[scl*0.0867,scl*-0.6000,0.0,1.0],
			[scl*0.1084,scl*-0.6723,0.0,1.0],
			[scl*0.1229,scl*-0.7446,0.0,1.0],
			[scl*0.1374,scl*-0.8169,0.0,1.0],
			[scl*0.1518,scl*-0.8892,0.0,1.0],
			[scl*0.1735,scl*-0.9615,0.0,1.0],
			[scl*0.1880,scl*-1.0337,0.0,1.0],
			[scl*0.2096,scl*-1.1060,0.0,1.0],
			[scl*0.2241,scl*-1.1783,0.0,1.0],
			[scl*0.2386,scl*-1.2506,0.0,1.0],
			[scl*0.2458,scl*-1.3229,0.0,1.0],
			[scl*0.2602,scl*-1.3952,0.0,1.0],
			[scl*0.2675,scl*-1.4675,0.0,1.0],
			[scl*0.2747,scl*-1.5398,0.0,1.0],
			[scl*0.2747,scl*-1.6121,0.0,1.0],
			[scl*0.2747,scl*-1.6844,0.0,1.0],
			[scl*0.2747,scl*-1.7566,0.0,1.0],
			[scl*0.2819,scl*-1.8289,0.0,1.0],
			[scl*0.2819,scl*-1.9012,0.0,1.0],
			[scl*0.2819,scl*-1.9735,0.0,1.0],
			[scl*0.2819,scl*-2.0458,0.0,1.0],
			[scl*0.2892,scl*-2.1181,0.0,1.0],
			[scl*0.2892,scl*-2.1904,0.0,1.0],
			[scl*0.3036,scl*-2.2627,0.0,1.0],
			[scl*0.3253,scl*-2.3205,0.0,1.0]]))
		L3_vein = np.transpose(np.array([
			[scl*-0.0145,scl*-0.2241,0.0,1.0],
			[scl*0.0,scl*-0.3108,0.0,1.0],
			[scl*0.0145,scl*-0.3831,0.0,1.0],
			[scl*0.0145,scl*-0.4554,0.0,1.0],
			[scl*0.0072,scl*-0.5277,0.0,1.0],
			[scl*0.0072,scl*-0.6000,0.0,1.0],
			[scl*0.0,scl*-0.6723,0.0,1.0],
			[scl*0.0072,scl*-0.7157,0.0,1.0],
			[scl*0.0145,scl*-0.7446,0.0,1.0],
			[scl*0.0217,scl*-0.8169,0.0,1.0],
			[scl*0.0361,scl*-0.8892,0.0,1.0],
			[scl*0.0434,scl*-0.9615,0.0,1.0],
			[scl*0.0506,scl*-1.0337,0.0,1.0],
			[scl*0.0578,scl*-1.1060,0.0,1.0],
			[scl*0.0578,scl*-1.1783,0.0,1.0],
			[scl*0.0578,scl*-1.2506,0.0,1.0],
			[scl*0.0578,scl*-1.3229,0.0,1.0],
			[scl*0.0578,scl*-1.3952,0.0,1.0],
			[scl*0.0651,scl*-1.4675,0.0,1.0],
			[scl*0.0651,scl*-1.5398,0.0,1.0],
			[scl*0.0723,scl*-1.6121,0.0,1.0],
			[scl*0.0723,scl*-1.6844,0.0,1.0],
			[scl*0.0723,scl*-1.7566,0.0,1.0],
			[scl*0.0723,scl*-1.8289,0.0,1.0],
			[scl*0.0651,scl*-1.9012,0.0,1.0],
			[scl*0.0651,scl*-1.9735,0.0,1.0],
			[scl*0.0578,scl*-2.0458,0.0,1.0],
			[scl*0.0578,scl*-2.1181,0.0,1.0],
			[scl*0.0506,scl*-2.1904,0.0,1.0],
			[scl*0.0506,scl*-2.2627,0.0,1.0],
			[scl*0.0361,scl*-2.3350,0.0,1.0],
			[scl*0.0217,scl*-2.4073,0.0,1.0],
			[scl*0.0145,scl*-2.4795,0.0,1.0],
			[scl*0.0072,scl*-2.5518,0.0,1.0],
			[scl*0.0,scl*-2.6241,0.0,1.0]]))
		L4_vein = np.transpose(np.array([
			[scl*-0.0867,scl*-0.0145,0.0,1.0],
			[scl*-0.0651,scl*-0.0940,0.0,1.0],
			[scl*-0.0506,scl*-0.1663,0.0,1.0],
			[scl*-0.0723,scl*-0.2386,0.0,1.0],
			[scl*-0.0795,scl*-0.3108,0.0,1.0],
			[scl*-0.0795,scl*-0.3831,0.0,1.0],
			[scl*-0.0867,scl*-0.4554,0.0,1.0],
			[scl*-0.0867,scl*-0.5277,0.0,1.0],
			[scl*-0.0940,scl*-0.6000,0.0,1.0],
			[scl*-0.0867,scl*-0.6723,0.0,1.0],
			[scl*-0.0795,scl*-0.6940,0.0,1.0],
			[scl*-0.1012,scl*-0.7446,0.0,1.0],
			[scl*-0.1084,scl*-0.8169,0.0,1.0],
			[scl*-0.1301,scl*-0.8892,0.0,1.0],
			[scl*-0.1446,scl*-0.9615,0.0,1.0],
			[scl*-0.1518,scl*-1.0337,0.0,1.0],
			[scl*-0.1663,scl*-1.1060,0.0,1.0],
			[scl*-0.1807,scl*-1.1783,0.0,1.0],
			[scl*-0.1880,scl*-1.2506,0.0,1.0],
			[scl*-0.2024,scl*-1.2868,0.0,1.0],
			[scl*-0.1952,scl*-1.3229,0.0,1.0],
			[scl*-0.1880,scl*-1.3952,0.0,1.0],
			[scl*-0.1880,scl*-1.4675,0.0,1.0],
			[scl*-0.1880,scl*-1.5398,0.0,1.0],
			[scl*-0.1880,scl*-1.6121,0.0,1.0],
			[scl*-0.1880,scl*-1.6844,0.0,1.0],
			[scl*-0.1880,scl*-1.7566,0.0,1.0],
			[scl*-0.1880,scl*-1.8289,0.0,1.0],
			[scl*-0.1880,scl*-1.9012,0.0,1.0],
			[scl*-0.1952,scl*-1.9735,0.0,1.0],
			[scl*-0.1952,scl*-2.0458,0.0,1.0],
			[scl*-0.1952,scl*-2.1181,0.0,1.0],
			[scl*-0.1952,scl*-2.1904,0.0,1.0],
			[scl*-0.2024,scl*-2.2627,0.0,1.0],
			[scl*-0.2024,scl*-2.3350,0.0,1.0],
			[scl*-0.2096,scl*-2.4073,0.0,1.0],
			[scl*-0.2169,scl*-2.4795,0.0,1.0],
			[scl*-0.2386,scl*-2.5591,0.0,1.0]]))
		L5_vein = np.transpose(np.array([
			[scl*-0.0867,scl*-0.0145,0.0,1.0],
			[scl*-0.1229,scl*-0.0940,0.0,1.0],
			[scl*-0.1446,scl*-0.1663,0.0,1.0],
			[scl*-0.1663,scl*-0.2386,0.0,1.0],
			[scl*-0.1807,scl*-0.3108,0.0,1.0],
			[scl*-0.1952,scl*-0.3831,0.0,1.0],
			[scl*-0.2024,scl*-0.4554,0.0,1.0],
			[scl*-0.2241,scl*-0.5277,0.0,1.0],
			[scl*-0.2386,scl*-0.6000,0.0,1.0],
			[scl*-0.2602,scl*-0.6723,0.0,1.0],
			[scl*-0.2747,scl*-0.7446,0.0,1.0],
			[scl*-0.2892,scl*-0.8169,0.0,1.0],
			[scl*-0.3108,scl*-0.8892,0.0,1.0],
			[scl*-0.3253,scl*-0.9615,0.0,1.0],
			[scl*-0.3470,scl*-1.0337,0.0,1.0],
			[scl*-0.3615,scl*-1.1060,0.0,1.0],
			[scl*-0.3904,scl*-1.1783,0.0,1.0],
			[scl*-0.4048,scl*-1.2506,0.0,1.0],
			[scl*-0.4410,scl*-1.3229,0.0,1.0],
			[scl*-0.4988,scl*-1.4313,0.0,1.0],
			[scl*-0.5494,scl*-1.4675,0.0,1.0],
			[scl*-0.6217,scl*-1.5398,0.0,1.0],
			[scl*-0.7012,scl*-1.5976,0.0,1.0]]))
		C1_vein = np.transpose(np.array([
			[scl*0.0,scl*-2.6241,0.0,1.0],
			[scl*-0.0578,scl*-2.6241,0.0,1.0],
			[scl*-0.1374,scl*-2.6097,0.0,1.0],
			[scl*-0.1880,scl*-2.5880,0.0,1.0],
			[scl*-0.2386,scl*-2.5518,0.0,1.0]]))
		C2_vein = np.transpose(np.array([
			[scl*-0.2386,scl*-2.5518,0.0,1.0],
			[scl*-0.3181,scl*-2.4795,0.0,1.0],
			[scl*-0.3904,scl*-2.4073,0.0,1.0],
			[scl*-0.4410,scl*-2.3350,0.0,1.0],
			[scl*-0.4916,scl*-2.2627,0.0,1.0],
			[scl*-0.5277,scl*-2.1904,0.0,1.0],
			[scl*-0.5566,scl*-2.1181,0.0,1.0],
			[scl*-0.5928,scl*-2.0458,0.0,1.0],
			[scl*-0.6145,scl*-1.9735,0.0,1.0],
			[scl*-0.6362,scl*-1.9012,0.0,1.0],
			[scl*-0.6578,scl*-1.8289,0.0,1.0],
			[scl*-0.6795,scl*-1.7566,0.0,1.0],
			[scl*-0.7012,scl*-1.6844,0.0,1.0],
			[scl*-0.7012,scl*-1.6121,0.0,1.0],
			[scl*-0.7012,scl*-1.5976,0.0,1.0]]))
		C3_vein = np.transpose(np.array([
			[scl*-0.7012,scl*-1.5976,0.0,1.0],
			[scl*-0.7229,scl*-1.5398,0.0,1.0],
			[scl*-0.7446,scl*-1.4675,0.0,1.0],
			[scl*-0.7518,scl*-1.3952,0.0,1.0],
			[scl*-0.7663,scl*-1.3229,0.0,1.0],
			[scl*-0.7735,scl*-1.2506,0.0,1.0],
			[scl*-0.7807,scl*-1.1783,0.0,1.0],
			[scl*-0.7807,scl*-1.1060,0.0,1.0],
			[scl*-0.7807,scl*-1.0337,0.0,1.0],
			[scl*-0.7880,scl*-0.9615,0.0,1.0],
			[scl*-0.7880,scl*-0.8892,0.0,1.0],
			[scl*-0.7880,scl*-0.8169,0.0,1.0],
			[scl*-0.7807,scl*-0.7446,0.0,1.0],
			[scl*-0.7663,scl*-0.6723,0.0,1.0],
			[scl*-0.7590,scl*-0.6000,0.0,1.0],
			[scl*-0.7446,scl*-0.5277,0.0,1.0],
			[scl*-0.7229,scl*-0.4554,0.0,1.0],
			[scl*-0.6940,scl*-0.3831,0.0,1.0],
			[scl*-0.6434,scl*-0.3108,0.0,1.0],
			[scl*-0.5566,scl*-0.2386,0.0,1.0],
			[scl*-0.3831,scl*-0.1663,0.0,1.0],
			[scl*-0.2169,scl*-0.0940,0.0,1.0],
			[scl*-0.1157,scl*-0.0217,0.0,1.0],
			[scl*-0.0867,scl*-0.0145,0.0,1.0]]))
		A_vein = np.transpose(np.array([[scl*0.0072,scl*-0.7157,0.0,1.0],
			[scl*-0.0795,scl*-0.6940,0.0,1.0]]))
		P_vein = np.transpose(np.array([[scl*-0.1952,scl*-1.2868,0.0,1.0],
			[scl*-0.3325,scl*-1.2868,0.0,1.0],
			[scl*-0.4048,scl*-1.2578,0.0,1.0]]))
		q_norm = np.sqrt(pow(state_in[0],2)+pow(state_in[1],2)+pow(state_in[2],2)+pow(state_in[3],2))
		q_0 = np.array([[state_in[0]/q_norm],
			[state_in[1]/q_norm],
			[state_in[2]/q_norm],
			[state_in[3]/q_norm]])
		T = np.array([
			[state_in[4]],
			[state_in[5]],
			[state_in[6]]])
		b1 = state_in[7]/3.0
		b2 = b1
		b3 = b1
		R_0 = np.array([[2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[1],2), 2.0*q_0[1]*q_0[2]+2.0*q_0[0]*q_0[3],  2.0*q_0[1]*q_0[3]-2.0*q_0[0]*q_0[2]],
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
		# transform L0, L1, L2:
		M_0 = np.zeros((4,4))
		M_0[0:3,0:3] = np.squeeze(R_0)
		M_0[0:3,3] = np.squeeze(T)
		M_0[3,3] = 1.0
		L0_pts = np.dot(M_0,L0_vein)
		L1_pts = np.dot(M_0,L1_vein)
		L2_pts = np.dot(M_0,L2_vein)
		# transform L3 ,A and C1:
		M_1 = np.zeros((4,4))
		#M_1[0:3,0:3] = np.dot(np.squeeze(R_1),np.squeeze(R_0))
		M_1[0:3,0:3] = np.dot(np.squeeze(R_0),np.squeeze(R_1))
		M_1[0:3,3] = np.squeeze(T)
		M_1[3,3] = 1.0
		L3_pts = np.dot(M_1,L3_vein)
		A_pts = np.dot(M_1,A_vein)
		C1_pts = np.dot(M_1,C1_vein)
		# transform L4, C2 and P:
		M_2 = np.zeros((4,4))
		#M_2[0:3,0:3] = np.dot(np.squeeze(R_2),M_1[0:3,0:3])
		M_2[0:3,0:3] = np.dot(M_1[0:3,0:3],np.squeeze(R_2))
		M_2[0:3,3] = np.squeeze(T)
		M_2[3,3] = 1.0
		L4_pts = np.dot(M_2,L4_vein)
		C2_pts = np.dot(M_2,C2_vein)
		P_pts = np.dot(M_2,P_vein)
		# transform L5 and C3:
		M_3 = np.zeros((4,4))
		#M_3[0:3,0:3] = np.squeeze(np.dot(np.squeeze(R_3),M_2[0:3,0:3]))
		M_3[0:3,0:3] = np.dot(np.squeeze(M_2[0:3,0:3]),np.squeeze(R_3))
		M_3[0:3,3] = np.squeeze(T)
		M_3[3,3] = 1.0
		L5_pts = np.dot(M_3,L5_vein)
		C3_pts = np.dot(M_3,C3_vein)
		contour_list = [L0_pts,L1_pts,L2_pts,L3_pts,L4_pts,L5_pts,C1_pts,C2_pts,C3_pts,A_pts,P_pts]
		return contour_list

	def transform_wing_L(self,state_in,scale_in):
		key_pts_0 = np.array([
			[scale_in*0.2313, scale_in*0.3253, scale_in*0.0, scale_in*0.0072],
			[scale_in*0.5711, scale_in*2.3205, scale_in*2.6241, scale_in*0.7157],
			[0.0, 0.0, 0.0, 0.0],
			[1.0, 1.0, 1.0, 1.0]])
		key_pts_1 = np.array([
			[scale_in*-0.2386, scale_in*-0.1952, scale_in*-0.0867],
			[scale_in*2.5591,  scale_in*1.2868,  scale_in*0.0145],
			[0.0,  0.0,   0.0 ],
			[1.0,  1.0,   1.0 ]])
		key_pts_2 = np.array([
			[scale_in*-0.7012, scale_in*-0.4048],
			[scale_in*1.5976, scale_in*1.2578],
			[0.0,  0.0],
			[1.0,  1.0]])
		key_pts_3 = np.array([
			[scale_in*-0.7880],
			[scale_in*0.8892],
			[0.0],
			[1.0]])
		q_norm = np.sqrt(pow(state_in[0],2)+pow(state_in[1],2)+pow(state_in[2],2)+pow(state_in[3],2))
		q_0 = np.array([[state_in[0]],
			[state_in[1]],
			[state_in[2]],
			[state_in[3]]])
		T = np.array([
			[state_in[4]],
			[state_in[5]],
			[state_in[6]]])
		b1 = state_in[7]/3.0
		b2 = b1
		b3 = b1
		R_0 = np.array([[2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[1],2), 2.0*q_0[1]*q_0[2]+2.0*q_0[0]*q_0[3],  2.0*q_0[1]*q_0[3]-2.0*q_0[0]*q_0[2]],
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
		pts_0 = np.dot(M_0,key_pts_0)
		# transform key_pts_1:
		M_1 = np.zeros((4,4))
		M_1[0:3,0:3] = np.dot(np.squeeze(R_1),np.squeeze(R_0))
		M_1[0:3,3] = np.squeeze(T)
		M_1[3,3] = 1.0
		pts_1 = np.dot(M_1,key_pts_1)
		# transform key_pts_2:
		M_2 = np.zeros((4,4))
		M_2[0:3,0:3] = np.dot(np.squeeze(R_2),M_1[0:3,0:3])
		M_2[0:3,3] = np.squeeze(T)
		M_2[3,3] = 1.0
		pts_2 = np.dot(M_2,key_pts_2)
		# transform key_pts_3:
		M_3 = np.zeros((4,4))
		M_3[0:3,0:3] = np.squeeze(np.dot(np.squeeze(R_3),M_2[0:3,0:3]))
		M_3[0:3,3] = np.squeeze(T)
		M_3[3,3] = 1.0
		pts_3 = np.dot(M_3,key_pts_3)
		pts_out = np.zeros(31)
		pts_out[:30] = np.squeeze(np.reshape(np.concatenate((pts_0[0:3,:],pts_1[0:3,:],pts_2[0:3,:],pts_3[0:3,:]),axis=1),(1,30)))
		pts_out[30] = q_norm
		return pts_out

	def transform_wing_R(self,state_in,scale_in):
		key_pts_0 = np.array([
			[scale_in*0.2313, scale_in*0.3253, scale_in*0.0, scale_in*0.0072],
			[scale_in*-0.5711, scale_in*-2.3205, scale_in*-2.6241, scale_in*-0.7157],
			[0.0, 0.0, 0.0, 0.0],
			[1.0, 1.0, 1.0, 1.0]])
		key_pts_1 = np.array([
			[scale_in*-0.2386, scale_in*-0.1952, scale_in*-0.0867],
			[scale_in*-2.5591,  scale_in*-1.2868,  scale_in*-0.0145],
			[0.0,  0.0,   0.0 ],
			[1.0,  1.0,   1.0 ]])
		key_pts_2 = np.array([
			[scale_in*-0.7012, scale_in*-0.4048],
			[scale_in*-1.5976, scale_in*-1.2578],
			[0.0,  0.0],
			[1.0,  1.0]])
		key_pts_3 = np.array([
			[scale_in*-0.7880],
			[scale_in*-0.8892],
			[0.0],
			[1.0]])
		q_norm = np.sqrt(pow(state_in[0],2)+pow(state_in[1],2)+pow(state_in[2],2)+pow(state_in[3],2))
		q_0 = np.array([[state_in[0]],
			[state_in[1]],
			[state_in[2]],
			[state_in[3]]])
		T = np.array([
			[state_in[4]],
			[state_in[5]],
			[state_in[6]]])
		b1 = state_in[7]/3.0
		b2 = b1
		b3 = b1
		R_0 = np.array([[2.0*pow(q_0[0],2)-1.0+2.0*pow(q_0[1],2), 2.0*q_0[1]*q_0[2]+2.0*q_0[0]*q_0[3],  2.0*q_0[1]*q_0[3]-2.0*q_0[0]*q_0[2]],
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
		pts_0 = np.dot(M_0,key_pts_0)
		# transform key_pts_1:
		M_1 = np.zeros((4,4))
		M_1[0:3,0:3] = np.dot(np.squeeze(R_1),np.squeeze(R_0))
		M_1[0:3,3] = np.squeeze(T)
		M_1[3,3] = 1.0
		pts_1 = np.dot(M_1,key_pts_1)
		# transform key_pts_2:
		M_2 = np.zeros((4,4))
		M_2[0:3,0:3] = np.dot(np.squeeze(R_2),M_1[0:3,0:3])
		M_2[0:3,3] = np.squeeze(T)
		M_2[3,3] = 1.0
		pts_2 = np.dot(M_2,key_pts_2)
		# transform key_pts_3:
		M_3 = np.zeros((4,4))
		M_3[0:3,0:3] = np.squeeze(np.dot(np.squeeze(R_3),M_2[0:3,0:3]))
		M_3[0:3,3] = np.squeeze(T)
		M_3[3,3] = 1.0
		pts_3 = np.dot(M_3,key_pts_3)
		pts_out = np.zeros(31)
		pts_out[:30] = np.squeeze(np.reshape(np.concatenate((pts_0[0:3,:],pts_1[0:3,:],pts_2[0:3,:],pts_3[0:3,:]),axis=1),(1,30)))
		pts_out[30] = q_norm
		return pts_out

	'''

	def calculate_scale(self,pts_in):
		d_01_ref = np.sqrt(pow(pts_in[1,0]-pts_in[0,0],2)+pow(pts_in[1,1]-pts_in[0,1],2)+pow(pts_in[1,2]-pts_in[0,2],2))
		d_02_ref = np.sqrt(pow(pts_in[2,0]-pts_in[0,0],2)+pow(pts_in[2,1]-pts_in[0,1],2)+pow(pts_in[2,2]-pts_in[0,2],2))
		d_03_ref = np.sqrt(pow(pts_in[3,0]-pts_in[0,0],2)+pow(pts_in[3,1]-pts_in[0,1],2)+pow(pts_in[3,2]-pts_in[0,2],2))
		d_49_ref = np.sqrt(pow(pts_in[9,0]-pts_in[4,0],2)+pow(pts_in[9,1]-pts_in[4,1],2)+pow(pts_in[9,2]-pts_in[4,2],2))
		d_01_key = np.sqrt(pow(0.3253-0.2313,2)+pow(2.3205-0.5711,2))
		d_02_key = np.sqrt(pow(0.0-0.2313,2)+pow(2.6241-0.5711,2))
		d_03_key = np.sqrt(pow(-0.2386-0.2313,2)+pow(2.5591-0.5711,2))
		d_49_key = np.sqrt(pow(-0.0867+0.7012,2)+pow(0.0145-1.5976,2))
		scale_out = 0.25*(d_01_ref/d_01_key+d_02_ref/d_02_key+d_03_ref/d_03_key+d_49_ref/d_49_key)
		return scale_out

	def cost_func_L(self,x,y_pts):
		x_pts = self.transform_wing_L(x,self.scale_L)
		err = self.weights*(x_pts-y_pts)
		return err

	def cost_func_R(self,x,y_pts):
		x_pts = self.transform_wing_R(x,self.scale_R)
		err = self.weights*(x_pts-y_pts)
		return err

	def calc_beta(self,pts_in):
		# Calculate normals:
		P0 = pts_in[[0,1,2,8],:]
		pca0 = PCA()
		s0 = pca0.fit_transform(P0)
		x0 = pca0.components_[1,:]
		y0 = pca0.components_[0,:]
		z0 = pca0.components_[2,:]
		if s0[1,1] < 0.0:
			x0 = -x0
			y0 = -y0
			z0 = -z0
		P1 = pts_in[[2,3,9],:]
		pca1 = PCA()
		s1 = pca1.fit_transform(P1)
		x1 = pca1.components_[1,:]
		y1 = pca1.components_[0,:]
		z1 = pca1.components_[2,:]
		if s1[0,1] < 0.0:
			x1 = -x1
			y1 = -y1
			z1 = -z1
		P2 = pts_in[[3,4,9],:]
		pca2 = PCA()
		s2 = pca2.fit_transform(P2)
		x2 = pca2.components_[1,:]
		y2 = pca2.components_[0,:]
		z2 = pca2.components_[2,:]
		if s2[0,1] < 0.0:
			x2 = -x2
			y2 = -y2
			z2 = -z2
		P3 = pts_in[[4,5,9],:]
		pca3 = PCA()
		s3 = pca3.fit_transform(P3)
		x3 = pca3.components_[1,:]
		y3 = pca3.components_[0,:]
		z3 = pca3.components_[2,:]
		if s3[0,1] < 0.0:
			x3 = -x3
			y3 = -y3
			z3 = -z3
		# Calculate cos(beta):
		cos_beta_1 = np.absolute(np.dot(z0,z1))/(np.linalg.norm(z0)*np.linalg.norm(z1))
		cos_beta_2 = np.absolute(np.dot(z1,z2))/(np.linalg.norm(z1)*np.linalg.norm(z2))
		cos_beta_3 = np.absolute(np.dot(z2,z3))/(np.linalg.norm(z2)*np.linalg.norm(z3))
		# Calculate the sign:
		if np.dot(np.cross(z0,z1),y0)>0.0:
			sign_1 = 1.0
		else:
			sign_1 = -1.0
		if np.dot(np.cross(z1,z2),y0)>0.0:
			sign_2 = 1.0
		else:
			sign_2 = -1.0
		if np.dot(np.cross(z2,z3),y0)>0.0:
			sign_3 = 1.0
		else:
			sign_3 = -1.0
		print(sign_1)
		print(sign_2)
		print(sign_3)
		# Calculate angles:
		beta_1 = sign_1*np.arccos(cos_beta_1)
		beta_2 = sign_2*np.arccos(cos_beta_2)
		beta_3 = sign_3*np.arccos(cos_beta_3)
		beta_array = np.array([beta_1,beta_2,beta_3])
		return beta_array

	def state(self,ref_pts_L,ref_pts_R):
		pts_L = np.zeros(31)
		pts_R = np.zeros(31)
		#print('beta_L')
		#beta_L = self.calc_beta(ref_pts_L[:,0:3])
		#print(beta_L)
		#print((beta_L[0]/np.pi)*180.0)
		#print((beta_L[1]/np.pi)*180.0)
		#print((beta_L[2]/np.pi)*180.0)
		#print('beta_R')
		#beta_R = self.calc_beta(ref_pts_R[:,0:3])
		#print(beta_R)
		#print((beta_R[0]/np.pi)*180.0)
		#print((beta_R[1]/np.pi)*180.0)
		#print((beta_R[2]/np.pi)*180.0)
		pts_L[:30] = np.squeeze(np.reshape(np.transpose(ref_pts_L[:,0:3]),(1,30)))
		pts_L[30] = 1.0
		pts_R[:30] = np.squeeze(np.reshape(np.transpose(ref_pts_R[:,0:3]),(1,30)))
		pts_R[30] = 1.0
		s_L = least_squares(self.cost_func_L,self.init_state_L,loss='cauchy',f_scale=0.1,args=([pts_L]),bounds=([self.lower_bnds,self.upper_bnds]))
		s_R = least_squares(self.cost_func_R,self.init_state_R,loss='cauchy',f_scale=0.1,args=([pts_R]),bounds=([self.lower_bnds,self.upper_bnds]))
		self.state_L = s_L.x
		self.state_R = s_R.x
		return self.state_L, self.state_R

	'''

	def set_scale(self,scale_L_in,scale_R_in):
		self.scale_L = scale_L_in
		self.scale_R = scale_R_in

	def set_state(self,state_L_in,state_R_in):
		self.state_L = state_L_in
		self.state_R = state_R_in