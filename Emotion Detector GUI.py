import tkinter as tk
from tkinter import font
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog


def start_camera():
    global cap, start_button, other_button, stop_button
    canvas.pack()
    cap = cv2.VideoCapture(0)
    Live_button.pack_forget()
    Upload_button.pack_forget()
    backlable.pack_forget()
    update()
    
    
def update():
    global cap, canvas, photo, window
    ret, frame = cap.read()
    if ret:
        photo = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())
        canvas.create_image(0, 0, anchor=tk.NW, image=photo, tags='stream')
    if cap is not None:
        window.after(15, update)
        
        

def other_option(): # hena hykon function upload photo aw take photo
    # Placeholder for other option functionality
    window.filename = filedialog.askopenfilename(initialdir="/gui/images", title="Select a file",filetypes=(("png files", "*.png"),("jpg files", "*.jpg")))
    Live_button.pack_forget()
    Upload_button.pack_forget()
    backlable.place_forget()
    uploadedpic= tk.Label(window, text= window.filename).pack()
    uploaded =ImageTk.PhotoImage(Image.open(window.filename))
    piclable = tk.Label(image=uploaded).pack()
    




def stop_camera():
    global cap, start_button, other_button, stop_button
    cap.release()
    #stop_button.pack_forget()
    Live_button.pack()
    Upload_button.pack()
    camera_frame.pack_forget()
    backlable.place(relx=0, rely=0, relwidth=1, relheight=1)
    window.after(1000, lambda: camera_frame.tkraise())
    camera_frame.pack()
    window.focus_force()
    
  


def close_window():
    window.destroy()
    
    

if __name__ == '__main__':
    cap = None
    photo = None
    window = tk.Tk()
    window.geometry('1000x800')
    window.title("AI project")
    canvas = tk.Canvas(window,width = 450 , height= 700)
    
    
    #camera frame stuff
    
    camera_frame = tk.Frame(window)
    camera_frame.pack(side="left", fill="x", expand=True)
    Stop_button = tk.Button(camera_frame, text="Stop Camera", command=stop_camera, state="active")
    Stop_button.pack(side="bottom", fill="y",  expand=True)
    canvas = tk.Canvas(camera_frame, width=450, height=700)
    canvas.pack()
    
    
    # Other Frame Stuff
    other_frame = tk.Frame(window)
    other_frame.pack(side="right", fill="both", expand=True)
    #other_button = tk.Button(other_frame, text="Other Function", command=)
    #other_button.pack(side="bottom")
    
    #Swich Fram Stuff
    #page_switcher_frame = tk.Frame(window)
    #page_switcher_frame.pack(side="bottom", fill="x")
   
    
    #main page Stuff
    image = Image.open("1.jpg")
    new_size = (window.winfo_screenwidth(), window.winfo_screenheight())
    image = image.resize(new_size)
    backpic = ImageTk.PhotoImage(image)
    backlable= tk.Label(window, image = backpic)
    backlable.place(relx=0, rely=0, relwidth=1, relheight=1)
    
    #Live Button
    image1 = Image.open("2.jpg")
    image1 = image1.resize((80,60))
    photo = ImageTk.PhotoImage(image1)
    Live_button = tk.Button(window, image=photo , command=start_camera, width= 80, height= 30)
    Live_button.place(relx=0.5, rely=0.87, anchor="center")
    
    #Ubload Button
    image2 = Image.open("3.jpg")
    image2 = image2.resize((90,70))
    photo2 = ImageTk.PhotoImage(image2)
    Upload_button = tk.Button(window, image=photo2 , command=other_option, width= 90, height= 30)
    Upload_button.place(relx=0.5, rely= 0.93, anchor="center")    
    
    
    
    
    
    
    window.mainloop()