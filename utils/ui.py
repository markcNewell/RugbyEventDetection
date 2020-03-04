import matplotlib.plt as plt
import io
import cv2
import numpy as np


def plot_preditions_graph(ruck,maul,scrum,lineout):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	x = [0,1,2,3]

	ax.set_ylim(0, 1)
	ax.set_xlabel("frame number")
	ax.set_ylabel("prediction")

	ax.plot(x, ruck, label="ruck")
	ax.plot(x, maul, label="maul")
	ax.plot(x, scrum, label="scrum")
	ax.plot(x, lineout, label="lineout")

	ax.legend()
	ax.set_title("Preditions")

	return fig




def overlay_images(overlay, base_image):
	x_offset=10
	y_offset=10
	base_image[y_offset:y_offset+overlay.shape[0], x_offset:x_offset+overlay.shape[1]] = overlay

	return base_image




def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img








