
import os
import tkinter as tk
import threading
import cv2 
import numpy as np
import funcs

from tkinter import filedialog, ttk
from PIL import Image, ImageTk

# ------------------------- Variables --------------------------

# ---------------- Images -----------------
tmp_img = None
original_img = None
style_img = None
styled_image = None
denoised_image = None
upscaled_image = None
image_to_save = None

output_list = None

# ---- Params, related to Vgg19 model ----
cnn, mean, std = funcs.load_vgg19()
name2idx, _ = funcs.get_name_idx_dicts(cnn)

# ---------- Training parameters ---------
num_steps = 300
style_weight = 1e6
content_weight = 1
content_layers = ['conv_22']
style_layers = ['conv_11', 'conv_12', 'conv_21', 'conv_22', 'conv_31']

# -------- ImgPreparer params -----------

preparer = None
from_gauss = False

# downacale params
apply_style_cmap = True
downscale_limit = 35000
do_blur = True
blur_params = {'ksize':(3, 3), 'sigmaX':1.5}

# upscale
resize_only=False
path_to_ESRGAN = 'RRDB_ESRGAN_x4.pth'
use_ESRGAN = 'first'

# remove_color_noise params
denoise = True
denoising_params = {
    'h': 5,
    'hColor': 2,
    'templateWindowSize': 5,
    'searchWindowSize': 10
}

# ------------------ Functions used for tab 1 ------------------

def load_image():
    global tmp_img
    file_path = filedialog.askopenfilename(title='Choose an image', 
                                            filetypes=[('Image files', '*.jpg;*.jpeg;*.png;')])
    if file_path:
        # Loading image
        img = Image.open(file_path)        
        # Convert PIL image to BGR numpy (opencv)
        tmp_img = np.array(img)
        tmp_img = tmp_img[:, :, ::-1].copy()
        return True
    return False


def load_original_img():
    global tmp_img, original_img, style_img
    if not load_image():
        return
    original_img = tmp_img.copy()

    img_tk = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_tk = Image.fromarray(img_tk)   
    img_tk.thumbnail((640, 480))
    img_tk = ImageTk.PhotoImage(image=img_tk)
    label_image_orig.config(image=img_tk)
    label_image_orig.image = img_tk

    if style_img is not None:
        reshape_style_label()
    
    
def load_style_img():
    global tmp_img, original_img, style_img
    if not load_image():
        return
    style_img = tmp_img.copy()

    img_tk = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
    img_tk = Image.fromarray(img_tk)
    img_tk.thumbnail((640, 480))   
    img_tk = ImageTk.PhotoImage(image=img_tk)
    label_image_style.config(image=img_tk)
    label_image_style.image = img_tk

    if original_img is not None:
        reshape_style_label()


def reshape_style_label():
    global original_img, style_img   
    root.update_idletasks()
    height_o = label_image_orig.winfo_height()
    width_s = style_img.shape[1] 
    height_s = style_img.shape[0]
    new_width = int(height_o*width_s/height_s)
    label_image_style.config(width=new_width, height=height_o)
    
    img_tk = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
    img_tk = cv2.resize(img_tk, (new_width, height_o))
    img_tk = Image.fromarray(img_tk) 
    img_tk = ImageTk.PhotoImage(image=img_tk)
    label_image_style.config(image=img_tk)
    label_image_style.image = img_tk

# ------------------ Functions used for tab 2 ------------------

# ------------ Checkboxes' functions --------------
def from_gauss_change():
    global from_gauss
    from_gauss = bool(from_gauss_var.get())

def resize_only_change():
    global resize_only
    resize_only = bool(resize_only_var.get())

def apply_style_cmap_change():
    global apply_style_cmap
    apply_style_cmap = bool(apply_style_cmap_var.get())

def do_blur_change():
    global do_blur
    do_blur = bool(do_blur_var.get())

def denoise_change():
    global denoise, denoised_image
    denoise = bool(denoise_var.get())
    resize_miniatures() 
        

# ------------ Radiobuttons' functions ------------

def select_esrgan_type():
    global use_ESRGAN
    val = use_ESRGAN_var.get()
    if val == 'None':
        val = None
    use_ESRGAN = val

# --------------- Entry functions -----------------

def validate_entry(new_value):
    if new_value == '':
        return True
    if new_value.isdigit():
        return True
    if new_value.count('.') <= 1:  
        if new_value.replace('.', '', 1).isdigit(): 
            return True
    return False 


def num_steps_entry_f():
    global num_steps
    new_value = int(float(num_steps_entry.get()))
    if new_value < 1:
        new_value = 1
    num_steps = new_value
    num_steps_label.config(text = f'Num Steps: {num_steps}')

def style_weight_entry_f():
    global style_weight
    new_value = float(style_weight_entry.get())
    style_weight = new_value
    style_weight_label.config(text = f'Style Weight: {style_weight}')

def content_weight_entry_f():
    global content_weight
    new_value = float(content_weight_entry.get())
    content_weight = new_value
    content_weight_label.config(text = f'Content Weight: {content_weight}')

def downscale_limit_entry_f():
    global downscale_limit
    new_value = int(float(downscale_limit_entry.get()))
    if new_value < 128*128:
        new_value = 128*128
    downscale_limit = new_value
    downscale_limit_label.config(text = f'Downscale Limit: {downscale_limit}')

def ksize_entry_f():
    global blur_params
    new_value = int(float(ksize_entry.get()))
    if new_value < 1:
        new_value = 1
    blur_params['ksize'] = (new_value, new_value)
    ksize_label.config(text = f'ksize: {blur_params['ksize']}')
    
def sigmax_entry_f():
    global blur_params
    new_value = float(sigmax_entry.get())
    if new_value < 0.1:
        new_value = 0.1
    blur_params['sigmaX'] = new_value
    sigmax_label.config(text = f'SigmaX: {denoising_params['h']}')


def h_entry_f():
    global denoising_params
    new_value = float(h_entry.get())
    if new_value < 0.1:
        new_value = 0.1
    denoising_params['h'] = new_value
    h_label.config(text = f'h: {denoising_params['h']}')
    resize_miniatures() 

def hcolor_entry_f():
    global denoising_params
    new_value = float(hcolor_entry.get())
    if new_value < 0.1:
        new_value = 0.1
    denoising_params['hColor'] = new_value
    hcolor_label.config(text = f'hColor: {denoising_params['hColor']}')
    resize_miniatures() 

def template_ws_entry_f():
    global denoising_params
    new_value = int(float(template_ws_entry.get()))
    if new_value < 1:
        new_value = 1
    denoising_params['templateWindowSize'] = new_value
    template_ws_label.config(text = f'Template Window Size: {denoising_params['templateWindowSize']}')
    resize_miniatures() 

def search_ws_entry_f():
    global denoising_params
    new_value = int(float(search_ws_entry.get()))
    if new_value < 1:
        new_value = 1
    denoising_params['searchWindowSize'] = new_value
    search_ws_label.config(text = f'Search Window Size: {denoising_params['searchWindowSize']}')
    resize_miniatures() 

# ------------------ Functions used for tab 3 ------------------

def run_transfer():
    global cnn, mean, std
    global num_steps, style_weight, content_weight
    global preparer, denoising_paramsm
    global downscale_limit, apply_style_cmap, from_gauss
    global styled_image, denoised_image, output_list, image_to_save
    global style_layers, content_layers

    if (original_img is None) or (style_img is None):
        tk.messagebox.showinfo(
            'Warning',
            f'Please, load initial and style images on the first tab'
        )
        return
    
    if len(content_layers) + len(style_layers) < 2:
        tk.messagebox.showinfo(
            'Warning',
            f'Please, choose at least one content and one style layer on settings tab'
        )
        return
    
    disable_widgets()
    preparer = funcs.ImgPreparer(original_img, style_img)
    content_img, style_img_, input_img = preparer.prepare_imgs(
        start_from_gauss=from_gauss, downscale_limit=downscale_limit, 
        apply_style_cmap=apply_style_cmap
    )
    output, output_list = funcs.run_style_transfer(
        cnn, mean, std,
        content_img, style_img_, input_img, 
        num_steps=num_steps, 
        style_weight=style_weight, content_weight=content_weight,
        style_layers=style_layers, content_layers=content_layers
    )
    enable_widgets()
    styled_image = preparer.restore_img(output)
    image_to_save = styled_image.copy()
    img_tk = cv2.cvtColor(styled_image, cv2.COLOR_BGR2RGB)
    
    if denoise:
        denoised_image = preparer.remove_color_noise(styled_image, denoising_params)
        image_to_save = denoised_image.copy()
        img_tk = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    
    ol = []
    for out in output_list:
        out = preparer.restore_img(out)
        ol.append(out)
    output_list = ol.copy()
    slider.config(to=len(output_list)-1)
    on_slider_change(slider.get())

    if resize_output.get():
        img_tk = preparer.upscale(img_tk, resize_only=True)
    img_tk = Image.fromarray(img_tk)
    img_tk.thumbnail((640, 480))   
    img_tk = ImageTk.PhotoImage(image=img_tk)
    label_image_result.config(image=img_tk)
    label_image_result.image = img_tk

def disable_widgets():
    for tab_id in notebook.tabs():
        tab_widget = notebook.nametowidget(tab_id)
        disable_children(tab_widget)
        
def disable_children(frame):
    for widget in frame.winfo_children(): 
        if isinstance(widget, 
                        (tk.Button, tk.Entry, 
                        tk.Checkbutton, tk.Radiobutton)
                        ):
            widget.config(state='disabled')
        elif isinstance(widget, tk.Frame):
            disable_children(widget)

def enable_widgets():
    for tab_id in notebook.tabs():
        tab_widget = notebook.nametowidget(tab_id)
        enable_children(tab_widget)

def enable_children(frame):
    for widget in frame.winfo_children(): 
        if isinstance(widget, 
                        (tk.Button, tk.Entry, 
                        tk.Checkbutton, tk.Radiobutton)
                        ):
            widget.config(state='normal')
        elif isinstance(widget, tk.Frame):
            enable_children(widget)

def start_transfer():
    thread = threading.Thread(target=run_transfer)
    thread.start()

def on_slider_change(idx):
    global output_list
    if output_list is not None:
        #idx = slider.get()
        img = output_list[int(idx)]
        if resize_output.get():
            img = preparer.upscale(img, resize_only=True)
        img_tk = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tk = Image.fromarray(img_tk)
        img_tk.thumbnail((640, 480))   
        img_tk = ImageTk.PhotoImage(image=img_tk)
        label_image_progress.config(image=img_tk)
        label_image_progress.image = img_tk

def resize_miniatures():
    global output_list, denoised_image

    if styled_image is not None:
        img = styled_image.copy()

        if denoise:
            denoised_image = preparer.remove_color_noise(img, denoising_params)
            img = denoised_image.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize_output.get():
            img = preparer.upscale(img, resize_only=True)
        img_tk = Image.fromarray(img)
        img_tk.thumbnail((640, 480))   
        img_tk = ImageTk.PhotoImage(image=img_tk)
        label_image_result.config(image=img_tk)
        label_image_result.image = img_tk
    
    on_slider_change(slider.get())    

# ------------------ Functions used for tab 4 ------------------

def save_path_to_ESRGAN():
    global path_to_ESRGAN
    file_path = filedialog.askopenfilename(
        title='Choose ESRGAN model weights',  defaultextension='.pth', 
        filetypes=[('PyTorch Model weights', '*.pth;*.pt')])
    if file_path: 
        path_to_ESRGAN = file_path


def run_upscale():
    global styled_image, denoised_image, upscaled_image
    global output, image_to_save, denoise
    global resize_only, path_to_ESRGAN, use_ESRGAN 

    if (use_ESRGAN is not None) and not (os.path.exists(path_to_ESRGAN)):
        tk.messagebox.showinfo(
            'Warning',
            f'ESRGAN weights not found at: {path_to_ESRGAN}. '
            'Please change upscale setting use_ESRGAN to None or '
            'load proper weights'
        )
        return
    if preparer is None:
        tk.messagebox.showinfo(
            'Warning',
            f'Image stylization was not done. '
            'Please run the style Transfer on tab 2'
        )
        return
    disable_widgets()
    img = styled_image.copy()
    if denoise:
        if denoised_image is not None:
            img = denoised_image
        else: 
            img = preparer.remove_color_noise(styled_image, denoising_params)

    upscaled_image = preparer.upscale(
        img, resize_only=resize_only, 
        use_ESRGAN=use_ESRGAN, 
        ESRGAN_weights_path=path_to_ESRGAN
    )
    enable_widgets()
    image_to_save = upscaled_image.copy()
    img_tk = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB)
    img_tk = Image.fromarray(img_tk)
    img_tk.thumbnail((640, 480))   
    img_tk = ImageTk.PhotoImage(image=img_tk)
    label_image_upscaled.config(image=img_tk)
    label_image_upscaled.image = img_tk

def start_upscale():
    thread = threading.Thread(target=run_upscale)
    thread.start()
# ------------------ Functions used for tab 5 ------------------

def save_image():
    global image_to_save
    if image_to_save is None:
        tk.messagebox.showinfo(
            'Warning',
            f'No images to save yet'
        )
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png", 
        filetypes=[("PNG files", "*.png"), 
                   ("JPEG files", "*.jpg"), 
                   ("All files", "*.*")]
        )
    if file_path:  
        img = cv2.cvtColor(image_to_save, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.save(file_path)  

#------------------------ Window  ------------------------------- 
root = tk.Tk()
root.title('Style Transfer')

notebook = ttk.Notebook(root)

#------------------------ Tab 1 elements ------------------------ 

tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='Image upload')

label1 = tk.Label(tab1, text='Upload images for the style transfer')
label1.pack(padx=20, pady=20)

frame_start_images = tk.Frame(tab1)
frame_start_images.pack(padx=20, pady=20)

load_button_orig = tk.Button(
    frame_start_images, 
    text='Load image to style', 
    command=load_original_img
)
load_button_orig.grid(row=0, column=0, padx=10, pady=5)

load_button_style = tk.Button(
    frame_start_images, 
    text='Load image to get the style from', 
    command=load_style_img
)
load_button_style.grid(row=0, column=1, padx=10, pady=5)

label_image_orig = tk.Label(frame_start_images)
label_image_orig.grid(row=1, column=0, padx=10, pady=5)

label_image_style = tk.Label(frame_start_images)
label_image_style.grid(row=1, column=1, padx=10, pady=5)

#------------------------ Tab 2 elements ------------------------

tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='Style Transfer Settings')

# ------------ Checkboxes  --------------

frame_chb = tk.Frame(tab2)
frame_chb.grid(row=0, column=3, sticky='n')

label_processing = tk.Label(frame_chb, text='Image procesisng options')
label_processing.grid(row=0, column=0)

do_blur_var = tk.IntVar(value=1)
do_blur_checkbox = tk.Checkbutton(
    frame_chb, text='Use GaussBlur before downscaling to remove noise', 
    variable=do_blur_var, command=do_blur_change 
)
do_blur_checkbox.grid(row=2, column=0, padx=5, pady=5, columnspan=2, sticky='w')

from_gauss_var = tk.IntVar(value=0)
from_gauss_checkbox = tk.Checkbutton(
    frame_chb, text='Start forming stylized image from noise instead of the original image', 
    variable=from_gauss_var, command=from_gauss_change 
)
from_gauss_checkbox.grid(row=1, column=0, padx=5, pady=5, columnspan=2, sticky='w')

apply_style_cmap_var = tk.IntVar(value=1)
apply_style_cmap_checkbox = tk.Checkbutton(
    frame_chb, text='Apply colormap of original image before Style Transfer', 
    variable=apply_style_cmap_var, command=apply_style_cmap_change 
)
apply_style_cmap_checkbox.grid(row=3, column=0, padx=5, pady=5, columnspan=2, sticky='w')

denoise_var = tk.IntVar(value=1)
denoise_checkbox = tk.Checkbutton(
    frame_chb, text='Remove color noise form stylized image before upscaling', 
    variable=denoise_var, command=denoise_change 
)
denoise_checkbox.grid(row=4, column=0, padx=5, pady=5, columnspan=2, sticky='w')

resize_only_var = tk.IntVar(value=0)
resize_only_checkbox = tk.Checkbutton(
    frame_chb, text='Do not upscale stylized image, only return to original size', 
    variable=resize_only_var, command=resize_only_change 
)
resize_only_checkbox.grid(row=5, column=0, padx=5, pady=5, columnspan=2, sticky='w')

# ------------ Radiobuttons  ------------

use_ESRGAN_var = tk.StringVar(value='first') 

frame_rb = tk.Frame(tab2)
frame_rb.grid(row=1, column=3, sticky='nw')

label_esrgan_type = tk.Label(frame_rb, text='Choose upscaling method')
label_esrgan_type.grid(row=0, column=0)

radio_es_1 = tk.Radiobutton(
    frame_rb, text='Use ESRGAN first, then Laplacian Pyramids', 
    variable=use_ESRGAN_var, value='first', 
    command=select_esrgan_type)
radio_es_1.grid(row=1, column=0, padx=5, pady=5, sticky='w')

radio_es_2 = tk.Radiobutton(
    frame_rb, text='Use Laplacian Pyramids first, then ESRGAN', 
    variable=use_ESRGAN_var, value='last', 
    command=select_esrgan_type)
radio_es_2.grid(row=2, column=0, padx=5, pady=5, sticky='w')

radio_es_3 = tk.Radiobutton(
    frame_rb, text='Use ESRGAN only', 
    variable=use_ESRGAN_var, value='all', 
    command=select_esrgan_type) 
radio_es_3.grid(row=3, column=0, padx=5, pady=5, sticky='w')

radio_es_4 = tk.Radiobutton(
    frame_rb, text='Use Laplacian Pyramids only', 
    variable=use_ESRGAN_var, value='None', 
    command=select_esrgan_type) 
radio_es_4.grid(row=4, column=0, padx=5, pady=5, sticky='w')

# --------------- Entries -----------------

frame_en= tk.Frame(tab2)
frame_en.grid(row=0, column=2, rowspan=3, sticky='n')

vcmd = (root.register(validate_entry), '%P')

label_num_params = tk.Label(frame_en, text='Tune numerical parameters')
label_num_params.grid(row=0, column=0, columnspan=3)

labels_and_entries = [
    ('Num Steps: 300', 'num_steps_entry', num_steps_entry_f),
    ('Style Weight: 1000000', 'style_weight_entry', style_weight_entry_f),
    ('Content Weight: 1', 'content_weight_entry', content_weight_entry_f),
    ('Downscale Limit: 35000', 'downscale_limit_entry', downscale_limit_entry_f),
    ('ksize: (3 x 3)', 'ksize_entry', ksize_entry_f),
    ('SigmaX: 1.5', 'sigmax_entry', sigmax_entry_f),
    ('h: 5', 'h_entry', h_entry_f),
    ('hColor: 2', 'hcolor_entry', hcolor_entry_f),
    ('Template Window Size: 5', 'template_ws_entry', template_ws_entry_f),
    ('Search Window Size: 10', 'search_ws_entry', search_ws_entry_f),
]

offset = 0
extra_labels = [
    'Transfer prameters', 
    'Gaussian Blur parameters', 
    'Color Denoise parameters'
] 
for i, (label_text, entry_name, func) in enumerate(labels_and_entries):
    if i in [0, 4, 6]:
        extra_label = tk.Label(frame_en, text=extra_labels[offset])
        extra_label.grid(row=i+1+offset, column=0, columnspan=3)
        offset +=1

    label = tk.Label(frame_en, text=label_text)
    label.grid(row=i+1+offset, column=0, sticky='w', padx=5, pady=5)

    entry = tk.Entry(frame_en, validate='key', validatecommand=vcmd)
    entry.grid(row=i+1+offset, column=1, padx=5, pady=5)

    button = tk.Button(frame_en, text='Submit', command=func)
    button.grid(row=i+1+offset, column=2, padx=5, pady=5)
    
    globals()[entry_name[:-5] + 'label'] = label
    globals()[entry_name] = entry

# --------------- Layer chboxes -----------------
frame_cb1 = tk.Frame(tab2)
frame_cb1.grid(row=0, column=0, rowspan=3, sticky='n')

label_cb1 = tk.Label(frame_cb1, text='Choose content layers\nfor Loss Calculation')
label_cb1.grid(row=0, column=0)

i = 0
cb1_vars = dict()
for name in name2idx:
    if 'conv' not in name:
        continue
    value = 0
    if name in content_layers:
        value = 1
    cb1_vars[name] = tk.IntVar(value=value)

    def var_changed(name=name):
        if cb1_vars[name].get() == 1:
            content_layers.append(name)
            content_layers.sort()
        elif name in content_layers:
            content_layers.remove(name)            

    conv_checkbox = tk.Checkbutton(
        frame_cb1, text=name, 
        variable=cb1_vars[name], command=var_changed 
    )
    conv_checkbox.grid(row=i+1, column=0, padx=5, pady=5)
    i += 1


frame_cb2 = tk.Frame(tab2)
frame_cb2.grid(row=0, column=1, rowspan=3, sticky='n')

label_cb2 = tk.Label(frame_cb2, text='Choose style layers\nfor Loss Calculation')
label_cb2.grid(row=0, column=0)

i = 0
cb2_vars = dict()
for name in name2idx:
    if 'conv' not in name:
        continue
    value = 0
    if name in style_layers:
        value = 1
    cb2_vars[name] = tk.IntVar(value=value)

    def var_changed(name=name):
        if cb2_vars[name].get() == 1:
            style_layers.append(name)
            style_layers.sort()
        elif name in style_layers:
            style_layers.remove(name)

    conv_checkbox = tk.Checkbutton(
        frame_cb2, text=name, 
        variable=cb2_vars[name], command=var_changed 
    )
    conv_checkbox.grid(row=i+1, column=0, padx=5, pady=5)
    i += 1

#------------------------ Tab 3 elements ------------------------

tab3 = ttk.Frame(notebook)
notebook.add(tab3, text='Style Transfer Results')

label3 = tk.Label(tab3, text='Run style transfer and see the results')
label3.grid(row=0, column=0, padx=10, pady=5, columnspan=2)

run_button = tk.Button(
    tab3, 
    text='Run Style Transfer', 
    command=start_transfer
)
run_button.grid(row=2, column=0, padx=10, pady=5)

label_image_result = tk.Label(tab3)
label_image_result.grid(row=3, column=0, padx=10, pady=5)

slider = tk.Scale(
    tab3, 
    from_=0, to=0, 
    orient='horizontal', length=300,
    command=on_slider_change)
slider.grid(row=2, column=1, padx=10, pady=5)

label_image_progress = tk.Label(tab3)
label_image_progress.grid(row=3, column=1, padx=5, pady=5)

resize_output = tk.IntVar(value=0)
resize_miniatures_checkbox = tk.Checkbutton(
    tab3, text='Resize outputs', 
    variable=resize_output, command=resize_miniatures
)
resize_miniatures_checkbox.grid(row=1, column=0, padx=5, pady=5, columnspan=2)
#------------------------ Tab 4 elements ------------------------

tab4 = ttk.Frame(notebook)
notebook.add(tab4, text='Upscaling')

esrgan_button = tk.Button(
    tab4, 
    text='Load ESRGAN model weights', 
    command=save_path_to_ESRGAN
)
esrgan_button.grid(row=0, column=0, padx=10, pady=5, sticky='nw')

upscale_button = tk.Button(
    tab4, 
    text='Upscale the results', 
    command=start_upscale
)
upscale_button.grid(row=1, column=0, padx=10, pady=5, sticky='nw')

label_image_upscaled = tk.Label(tab4)
label_image_upscaled.grid(row=0, column=1, padx=10, pady=5, rowspan=2)

#------------------------ Tab 5 elements ------------------------

tab5 = ttk.Frame(notebook)
notebook.add(tab5, text='Saving')


save_button = tk.Button(
    tab5, 
    text='Save last produced image', 
    command=save_image
)
save_button.grid(row=0, column=0, padx=10, pady=5)

#------------------------ End  -------------------------------

notebook.pack(expand=True, fill='both')

# Run window
root.mainloop()


