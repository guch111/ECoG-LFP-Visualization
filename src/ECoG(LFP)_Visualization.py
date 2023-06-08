import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from load_intan_rhd_format import read_data
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo, showerror
from scipy import signal
import os
from math import ceil


def run():
    cfg = cfg_load()
    if not cfg:
        return 0

    waveform = data_process(cfg)
    if type(waveform) is np.array:
        return 0

    if cfg['specgram_cfg']['plot_en']:
        if cfg['specgram_cfg']['NFFT'] <= cfg['specgram_cfg']['overlap']:
            showerror(title="ERROR", message="Specgram_cfg: overlap should be less than NFFT!")
            return 0
        if cfg['specgram_cfg']['db_range']:
            if (len(cfg['specgram_cfg']['db_range']) != 2 or 
                cfg['specgram_cfg']['db_range'][0] >= cfg['specgram_cfg']['db_range'][1]):
                showerror(title='ERROR', message='Wrong format in "specgram_cfg"-"db_range"!')
                return 0
        spec_plot(waveform, cfg)

    if cfg['waveform_plt_cfg']['plot_en']:
        sample_rate = cfg['data_process_cfg']['resample_freq']
        time_step = cfg['waveform_plt_cfg']['time_step']
        if cfg['waveform_plt_cfg']['time_win']:
            for w in cfg['waveform_plt_cfg']['time_win']:
                if not len(w)==2:
                    showerror(title="ERROR", message='Wrong format in "waveform_plt_cfg"-"time_win"!')
                else:
                    waveform_plt_win(cfg, waveform[:, w[0]*sample_rate:w[1]*sample_rate], 
                                    os.path.join(cfg['work_dir'], "%d-%dseconds.png"%(w[0], w[1])), w)
        else:
            fignum = ceil(waveform.shape[1]/sample_rate/time_step)
            for i in range(fignum):
                waveform_plt_win(cfg, waveform[:, i*time_step*sample_rate:(i+1)*time_step*sample_rate], 
                                os.path.join(cfg['work_dir'], "%d.png"%i),
                                [i*time_step,(i+1)*time_step])
    showinfo(title="", message="Complete!")

def cfg_load():
    cur_path = os.getcwd()
    if not os.path.exists(os.path.join(cur_path, 'plot_cfg.json')):
        showerror(title="ERROR", message="Missing plot_cfg.json file!")
        return 0
    with open(os.path.join(cur_path, 'plot_cfg.json'),'r') as cfg_f:
        cfg = json.load(cfg_f)
    if not cfg['version'] == 1.0:
        showerror(title="ERROR", message="Error version of plot_cfg.json file!")
        return 0
    cfg['filename'] = filename.get().replace("/","\\")
    cfg['data_process_cfg']['ch_list'] = list(range(cfg['data_process_cfg']['channel'][0], 
                                                    cfg['data_process_cfg']['channel'][1]+1))
    cfg['work_dir'] = os.path.dirname(cfg['filename'])
    return cfg

def data_process(cfg):
    if not cfg['filename']:
        showerror(title="ERROR", message='Do not select a RHD file!')
        return 0
    result, cfg['sample_rate'] = read_data(cfg['filename'])

    disable_ch = list(range(128))
    for ch_info in result['amplifier_channels']:
        disable_ch.remove(ch_info['native_order'])
    disable_ch.sort()
    data = result['amplifier_data']
    zero_ch = np.array([0]*(data.shape[1]))
    for c in disable_ch:
        data = np.insert(data, c, zero_ch, axis=0)

    data_sel = data[cfg['data_process_cfg']['channel'][0]: cfg['data_process_cfg']['channel'][1]+1, :]
    data_sel = data_sel/1000

    if cfg['data_process_cfg']['resample_freq'] < 200:
        showerror(title="ERROR", message="Resample freqence should not be lower than 200Hz!")
        return 0
    resample_len = int(data_sel.shape[1]*cfg['data_process_cfg']['resample_freq']/cfg['sample_rate'])
    waveform = signal.resample(data_sel, resample_len, axis=1)

    if cfg['data_process_cfg']['notch_filter_en']:
        w0 = cfg['data_process_cfg']['notch_filter_freq']/(cfg['data_process_cfg']['resample_freq']/2)
        if not (cfg['data_process_cfg']['notch_filter_freq']==50 or cfg['data_process_cfg']['notch_filter_freq']==60):
            showerror(title="ERROR", message="Notch filter frequence can only be 50Hz or 60Hz!")
            return 0
        b,a = signal.iirnotch(w0,30)
        waveform = signal.filtfilt(b,a,waveform, axis = 1)

    if cfg['data_process_cfg']['lowpass_filter_en']:
        if cfg['data_process_cfg']['lowpass_filter_freq'] <= 0: 
            showerror(title="ERROR", message="Resample frequence should not be lower than 0Hz!")
            return 0
        if cfg['data_process_cfg']['lowpass_filter_freq'] >= cfg['data_process_cfg']['resample_freq']/2: 
            showerror(title="ERROR", message="As resample frequence is {}Hz, lowpass filter frequence should be less than {}Hz!"
                      .format(cfg['data_process_cfg']['resample_freq'], cfg['data_process_cfg']['resample_freq']/2))
            return 0
        wn = 2*cfg['data_process_cfg']['lowpass_filter_freq']/cfg['data_process_cfg']['resample_freq']
        b,a = signal.butter(4, wn, 'lowpass')
        waveform = signal.filtfilt(b,a,waveform, axis = 1)

    if cfg['data_process_cfg']['highpass_filter_en']:
        if cfg['data_process_cfg']['highpass_filter_freq'] <= 0: 
            showerror(title="ERROR", message="Resample frequence should not be lower than 0Hz!")
            return 0
        if cfg['data_process_cfg']['highpass_filter_freq'] >= cfg['data_process_cfg']['resample_freq']/2: 
            showerror(title="ERROR", message="As resample frequence is {}Hz, highpass filter frequence should be less than {}Hz!"
                      .format(cfg['data_process_cfg']['resample_freq'], cfg['data_process_cfg']['resample_freq']/2))
            return 0
        wn = 2*cfg['data_process_cfg']['highpass_filter_freq']/cfg['data_process_cfg']['resample_freq']
        b,a = signal.butter(4, wn, 'highpass')
        waveform = signal.filtfilt(b,a,waveform, axis = 1)

    waveform = np.array(waveform)
    if cfg['data_process_cfg']['common_noise_remove_en']:
        common_noise = waveform.mean(axis=0)
        waveform -= common_noise
    return waveform

def spec_plot(waveform, cfg):
    for i in range(len(cfg['data_process_cfg']['ch_list'])):
        d = waveform[i,:]
        d = np.squeeze(d)
        plt.figure(figsize=(18, 8), dpi=150)
        if cfg['specgram_cfg']['db_range']:
            plt.specgram(d, NFFT=cfg['specgram_cfg']['NFFT'], Fs=cfg['data_process_cfg']['resample_freq'], 
                        noverlap=cfg['specgram_cfg']['overlap'], cmap='jet', Fc=0, 
                        vmin = cfg['specgram_cfg']['db_range'][0], 
                        vmax = cfg['specgram_cfg']['db_range'][1])
        else:
            plt.specgram(d, NFFT=cfg['specgram_cfg']['NFFT'], Fs=cfg['data_process_cfg']['resample_freq'], 
                        noverlap=cfg['specgram_cfg']['overlap'], cmap='jet', Fc=0)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title('Ch {} time_freq'.format(cfg['data_process_cfg']['ch_list'][i]))
        plt.colorbar(orientation="vertical")
        plt.savefig(os.path.join(cfg['work_dir'], "ch%d_specgram.png"%(cfg['data_process_cfg']['ch_list'][i])))
        plt.close()

def waveform_plt_win(cfg, waveform, path, win):
    data = waveform * cfg['waveform_plt_cfg']['amp_scale']
    samplerate = cfg['data_process_cfg']['resample_freq'] 
    chname = ["ch%d"%i for i in cfg['data_process_cfg']['ch_list']]
    ch_n, length = data.shape
    win_l = length/samplerate
    fig = plt.figure(dpi=cfg['waveform_plt_cfg']['dpi'], 
                     figsize=(1.5*length/samplerate, ch_n*0.5*cfg['waveform_plt_cfg']['spacing_scale']))
    ystep = 0.5*cfg['waveform_plt_cfg']['spacing_scale']
    ax = plt.axes()
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0, ystep*(ch_n-1), ch_n)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter([c for c in chname]))
    ax.set_xticks([])
    plt.tick_params(left=False)
    for i in ['top', 'right', 'bottom', 'left']:
        ax.spines[i].set_visible(False)
    for i in range(ch_n):
        y = data[i]+i*ystep
        x = np.linspace(0, length/samplerate, length)
        plt.plot(x, y, lw = cfg['waveform_plt_cfg']['linewidth'], c='black')
    plt.plot([win_l+0.4 for _ in range(2)], [0, 1*cfg['waveform_plt_cfg']['amp_scale']], lw=5, c='black')
    plt.text(win_l+0.8, 0.5*cfg['waveform_plt_cfg']['amp_scale'], '1mV', verticalalignment='center')
    plt.plot([win_l-0.8, win_l+0.2], [-1,-1], lw=5, c='black')
    plt.text(win_l-0.3, -1.5, '1s', horizontalalignment='center')
    plt.xlim(0, win_l+3)
    plt.ylim(-ystep*2, ystep*ch_n)
    plt.title("%d-%dseconds"%(win[0], win[1]))
    plt.savefig(path)
    plt.close(fig)

def file_select():
    filename.set('')
    tmp = askopenfilename()
    if tmp:
        filename.set(tmp.replace('/','\\'))
    

if __name__ == '__main__':
    root = tk.Tk()
    root.title('ECoG(LFP) Visualization V1.0')
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw-315) / 2
    y = (sh-65) / 2
    root.geometry('315x65+%d+%d'%(x,y))
    root.resizable(False, False)

    filename = tk.StringVar()
    tk.Entry(root, textvariable=filename, width=33).grid(row=0, column=0, sticky='w', columnspan=3, padx=5)
    tk.Button(root, text='选择文件', pady=0, font=('微软雅黑', 10), command=file_select).grid(row=0, column=3, sticky='w')

    tk.Button(root, text='开始', pady=0, font=('微软雅黑', 10), command=run).grid(row=1, column=0)

    root.mainloop()