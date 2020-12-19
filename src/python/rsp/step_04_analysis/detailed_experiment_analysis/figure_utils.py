import os

from rsp.step_04_analysis.plot_utils import PDF_HEIGHT
from rsp.step_04_analysis.plot_utils import PDF_WIDTH
from rsp.utils.file_utils import check_create_folder
from rsp.utils.rsp_logger import rsp_logger


def figure_show_or_save(fig, output_folder: str = None, file_name: str = None, width: int = PDF_WIDTH, height: int = PDF_HEIGHT):
    if output_folder is None and file_name is None:
        fig.show()
    else:
        if output_folder is None:
            output_folder = "."
        if file_name is None:
            file_name = f"figure.pdf"
        check_create_folder(output_folder)
        pdf_file = os.path.join(output_folder, file_name)
        # https://plotly.com/python/static-image-export/
        fig.write_image(pdf_file, width=width, height=height)
        rsp_logger.info(msg=f"wrote {pdf_file}")
