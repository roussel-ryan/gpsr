from image_processing import process_images
location = (
    "/global/homes/r/rroussel/phase_space_reconstruction/quad_scan_180A"
)
base_fname = location + "/DQ7_scan1_"

process_images(base_fname, 10)