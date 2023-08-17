import pyrender
import threading
import time
from loguru import logger
from urdf_test.utils.urdf import URDFLocal

robot = URDFLocal.load("data/ur5/ur5.urdf")
scene, nodes = robot.show(cfg={"shoulder_lift_joint": -2.0, "elbow_joint": 2.0})

viewer = None


def my_thread() -> None:
    global viewer
    count = 0
    while True:
        if viewer is None:
            time.sleep(0.1)
            logger.info("Waiting")
            continue

        viewer.render_lock.acquire()

        fk = robot.visual_trimesh_fk(cfg={"shoulder_lift_joint": -2.0 * (1 - count*0.01), "elbow_joint": count * 0.01})
        for tm, node in zip(fk, nodes):
            pose = fk[tm]
            scene.set_pose(node, pose)

        viewer.render_lock.release()

        time.sleep(0.05)
        count += 1


def createRunViewer() -> None:
    global viewer
    viewer = pyrender.Viewer(scene, auto_start=False, use_raymond_lighting=True, viewport_size=(1200, 1200))
    viewer.start()
    # viewer.ex


thr = threading.Thread(target=my_thread)
thr.start()

time.sleep(2)
createRunViewer()


# visualizer.start()
