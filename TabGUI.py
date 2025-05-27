from tkinter import *
import os
import os.path as osp

from h5_to_summary import GenerateSummaryFromH5
from src.models.CNN import ResNet, GoogleNet, Inception
from tkinter import ttk
import time

# import filedialog module
from tkinter import filedialog


# Function for opening the
# file explorer window
from tkinter.ttk import Progressbar

from src import *
from src.custom_dataset import Generate_Dataset
from summary2video import generateVideoSummary

lblProgress=None
def browseFiles():

    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("mp4",
                                                      "*.mp4"),
                                                     ("all files",
                                                      "*.*")))
    global lblProgress
    if(lblProgress==None):
        lblProgress = Label(GenerateDataSet, text='')
        lblProgress.grid(column=0, columnspan=4, row=1, padx=20, pady=10)
        GenerateDataSet.update()
    else:
        lblProgress.config(text='', foreground="black")
        GenerateDataSet.update()

    if(os.path.splitext(filename)[-1].lower()!='.mp4'):
        lblProgress.config(text='Select Valid mp4 file', foreground="red")
        GenerateDataSet.update()
        return
    progress = ttk.Progressbar(GenerateDataSet, length=200, cursor='spider',
                         mode="determinate",
                         orient=HORIZONTAL)
    progress.grid( column=5,columnspan=6, row=1,padx=20,pady=10)
    #try:
    outputname = f'dataset_custom_processed.h5'
    lblProgress.config(text='Setting up model ...')
    GenerateDataSet.update()
    gen = Generate_Dataset(filename, '', outputname, 'custom', progress, lblProgress, GenerateDataSet
            ,".\\datasets\\pytorch-i3d\\models\\flow_imagenet.pt",
             ".\\datasets\\pytorch-i3d\\models\\rgb_imagenet.pt",
             ".\\datasets\\3D-ResNets-PyTorch\\weights\\r3d101_KM_200ep.pth",
              ".\\frames",
       ".\\datasets\\object_features\\"
     )
    progress['value']=0
    GenerateDataSet.update()
    gen.generate_dataset()
    gen.h5_file.close()

    progress.grid_forget()
    lblProgress.config(text='DataSet Generated Successfully', foreground="Green")
    GenerateDataSet.update()
   # except:
    #    progress.grid_forget()
     #   lblProgress.config(text='DataSet Generation Failed', foreground="red")
     #   GenerateDataSet.update()
    print(filename)
# function to generate summary from h5 file
lblSummaryFrom=None
def GenerateSummaryForVideo():
    global lblSummaryFrom
    if(lblSummaryFrom==None):
        lblSummaryFrom = Label(GenerateSummary, text='Generating Summary for Dataset...')
        lblSummaryFrom.grid(column=0, row=1, padx=50, pady=20)
    else:
        lblSummaryFrom.config(text='Generating Summary for Dataset...', foreground="black")
        GenerateSummary.update()

    if not osp.exists(".\\datasets\\object_features\\dataset_custom_processed.h5"):
        lblSummaryFrom.config(text='Dataset does not exists', foreground="red")
        GenerateSummary.update()
        return
    try:
        GenerateSummaryFromH5()
        lblSummaryFrom.config(text='Summary Generated Succesfully', foreground="Green")
        GenerateSummary.update()
    except:
        lblSummaryFrom.config(text='Summary Generation Failed', foreground="red")
        GenerateSummary.update()


# function to generate Video from h5 file
lblGenerateVideo=None
def GenerateSummarizedVideo():
    global lblGenerateVideo
    if(lblGenerateVideo==None):
        lblGenerateVideo = Label(GenerateVideo, text='Generating video from Summary...')
        lblGenerateVideo.grid(column=0, row=1, padx=50, pady=20)
        GenerateVideo.update()
    else:
        lblGenerateVideo.config(text='Generating video from Summary...', foreground="black")
        GenerateVideo.update()

    if not osp.exists(".\\log\\result.h5"):
        lblGenerateVideo.config(text='Summarized h5 does not exists', foreground="red")
        GenerateVideo.update()
        return
    try:
        summaryProgress = ttk.Progressbar(GenerateVideo, length=200, cursor='spider',
                                   mode="determinate",
                                   orient=HORIZONTAL)
        summaryProgress.grid(column=0,  row=2, padx=50, pady=20)
        generateVideoSummary(videolabel,summaryProgress,GenerateVideo)
        lblGenerateVideo.config(text='Summarized video Generated Successfully', foreground="Green")
        #        lblGenerateVideo.grid_forget();
        summaryProgress.grid_forget()
        GenerateVideo.update()
    except:
        lblGenerateVideo.config(text='Summary to video Generation Failed', foreground="red")
        GenerateVideo.update()

window = Tk()
mygreen = "#d2ffd2"
myred = "#dd0202"

style = ttk.Style()

style.theme_create( "custom-style", parent="alt", settings={
        "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
        "TNotebook.Tab": {
            "configure": {"padding": [5, 1], "background": mygreen },
            "map":       {"background": [("selected", myred)],
                          "expand": [("selected", [1, 1, 1, 0])] } } } )

#style.theme_use("custom-style")

window.geometry('480x300')

window.title("Video Summarization Application")

tab_control = ttk.Notebook(window)

GenerateDataSet = ttk.Frame(tab_control)

GenerateSummary = ttk.Frame(tab_control)
GenerateVideo = ttk.Frame(tab_control)

tab_control.add(GenerateDataSet, text='Generate DataSet')

tab_control.add(GenerateSummary, text='Generate Summary from DataSet')
tab_control.add(GenerateVideo, text='Generate Video from Summary')

##################################
#          Generate DataSet
##################################
lblYourVideo = Label(GenerateDataSet, text= 'Select your video : ')#.place(x=300, y=400)
lblYourVideo.grid(column=0,columnspan=4, row=0,padx=20,pady=30)
button_explore = Button(GenerateDataSet,
                        text="Browse Files",
                        command=browseFiles)
button_explore.grid(column=5,columnspan=6, row=0,padx=20,pady=30)

##################################
#          Generate Summary
##################################

button_explore = Button(GenerateSummary,
                        text="Generate Summary for video",
                        command=GenerateSummaryForVideo)
button_explore.grid(column=0, row=0,pady=30,padx=50)



##################################
#          Generate Video
##################################
button_explore = Button(GenerateVideo,
                        text="Generate Summarized Video",
                        command=GenerateSummarizedVideo)
button_explore.grid(column=0, row=0,pady=30,padx=50)
videolabel = Label(GenerateVideo)
videolabel.grid(column=0, row=1,padx=30,pady=10)

tab_control.pack(expand=1, fill='both')
window.mainloop()
