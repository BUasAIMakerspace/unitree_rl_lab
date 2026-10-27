# usage: python3 mujoco_check.py default <push m/s>   |   python3 mujoco_check.py zero 0  (reproduces the offset bug)
import mujoco, numpy as np, yaml, onnxruntime as ort, sys
P='./'  # run from inside the policy folder
d=yaml.safe_load(open(P+'params/deploy.yaml.orig'))
M=np.array(d['joint_ids_map']); kp=np.array(d['stiffness']); kd=np.array(d['damping'])  # SDK order
q0p=np.array(d['default_joint_pos']); q0=np.zeros(29); q0[M]=q0p           # -> SDK order
OFFSET_DEFAULT = (sys.argv[1]!='zero') if len(sys.argv)>1 else True
push = float(sys.argv[2]) if len(sys.argv)>2 else 0.0
s=ort.InferenceSession(P+'exported/policy.onnx')
m=mujoco.MjModel.from_xml_path('../unitree_mujoco/unitree_robots/g1/scene_29dof.xml'  # adjust path)
m.opt.timestep=0.005
dd=mujoco.MjData(m)
assert m.nu==29
names=[m.joint(i+1).name for i in range(29)]
dd.qpos[2]=0.79; dd.qpos[3]=1; dd.qpos[7:]=q0
mujoco.mj_forward(m,dd)
last=np.zeros(29); log=[]
for step in range(int(20/0.02)):
    qw,qx,qy,qz=dd.qpos[3:7]
    R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
    g=R.T@np.array([0,0,-1.]); w=dd.qvel[3:6]
    q=dd.qpos[7:]; qd=dd.qvel[6:]
    obs=np.concatenate([w*0.2,g,(q-q0)[M],(qd[M])*0.05,last]).astype(np.float32)[None]
    a=s.run(None,{'obs':obs})[0][0]; last=a
    tgt=np.zeros(29); tgt[M]=a*0.25+(q0p if OFFSET_DEFAULT else 0)
    if push and abs(step*0.02-8.0)<1e-6: dd.qvel[0]+=push   # sudden base velocity, +x
    if push and abs(step*0.02-14.0)<1e-6: dd.qvel[0]-=push  # from front (backwards push)
    for _ in range(4):
        tau=kp*(tgt-dd.qpos[7:])-kd*dd.qvel[6:]
        dd.ctrl[:]=np.clip(tau,-m.actuator_ctrlrange[:,1],m.actuator_ctrlrange[:,1]) if m.actuator_ctrllimited.any() else tau
        mujoco.mj_step(m,dd)
    log.append([step*0.02,dd.qpos[0],dd.qpos[1],dd.qpos[2],np.degrees(np.arccos(np.clip(-g[2],-1,1))),np.abs(qd).max()])
L=np.array(log)
print('offset=%s push=%.1f'%('default' if OFFSET_DEFAULT else 'zero',push))
for t in [0,1,2,5,8,9,10,14,15,16,19.98]:
    r=L[np.argmin(abs(L[:,0]-t))]; print('t=%5.1f  x=%+.2f y=%+.2f z=%.2f tilt=%5.1f deg  max|qd|=%.1f'%tuple(r))
print('max drift from start: %.2f m'%np.hypot(L[:,1]-L[0,1],L[:,2]-L[0,2]).max(), ' min height %.2f'%L[:,3].min())
q=dd.qpos[7:]; print('final pose dev from default (SDK order, rad):',np.round(q-q0,2))
