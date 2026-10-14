"""Figure 1: the task (a), what an agent reads and writes (b), and who wrote where (c).

Panels a and b are vector illustration; panel c draws cache/network.json, which
src/network.py writes. All labels are editable text at 7 to 8 pt.
Icon source and licence: assets/icons/lucide/.
"""
import collections
import json
import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, PathPatch
from matplotlib.transforms import Affine2D
from svgpath2mpl import parse_path

from common import CACHE, FIGS, PREVIEW, ICONS as _ICONS
OUT=Path(FIGS)
ICONS=Path(_ICONS)
# Exactly the palette of figures A-C: ink for the subject, teal for reading, amber for writing, slate and rule for
# everything passive. No tinted grounds: the wiki is a thin outline, the task and the record stand on white.
INK, SLATE, RULE='#1a1a1a','#6b7b8c','#b8bfc6'
TEAL, AMBER='#3d9aa1','#d99b3f'
GROUND='#f2f3f4'

# Three columns: task loop 0-36, agent and wiki 39-129, network 122-178.
W_MM, H_MM=183, 77
NET=dict(x0=126, x1=178.5, y0=5, y1=51, shift=-3)   # shift moves the network, not its legend
# Short names of the ten tasks with most handles, in order of handles.
TASKS={'datausa-clothing-workforce':'clothing workforce','oecd-equity':'OECD equity',
       'datausa-sector61-state':'education sector','datausa-grocery-workforce':'grocery workforce',
       'datausa-cashiers-masters':'cashiers','datausa-construction-workforce':'construction',
       'ihme-cvd-deaths':'CVD deaths','datausa-poverty-county':'county poverty',
       'datausa-language-french':'French speakers','datausa-maids-wage':'maids wage'}
TASK_COLOURS=['#4c5fa8','#d99b3f','#5f9e6e','#c1698e','#8c5a1e','#3d9aa1','#c0504d','#8064a2','#9b9b3a','#6fa8dc']


def main():
    mpl.rcParams.update({'font.family':['Arial','Liberation Sans','DejaVu Sans'],'font.size':6,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':0.6,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(W_MM/25.4,H_MM/25.4),facecolor='white')
    ax=fig.add_axes([0,0,1,1],xlim=(-4.5,178.5),ylim=(77,0))
    ax.set_aspect('equal'); ax.axis('off')

    def text(x,y,s,size=6,bold=False,ha='left',color=INK,**kw):
        return ax.text(x,y,s,fontsize=size,weight='bold' if bold else 'normal',
                       ha=ha,va='center',color=color,linespacing=1.3,zorder=12,**kw)
    def box(x,y,w,h,fc='white',ec=INK,lw=.7,r=1,**kw):
        patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                            facecolor=fc,edgecolor=ec,lw=lw,**kw)
        ax.add_patch(patch)
    def arrow(p,q,c=INK,rad=0,lw=.8,dashed=False):
        ax.add_patch(FancyArrowPatch(p,q,connectionstyle=f'arc3,rad={rad}',
                     arrowstyle='->',mutation_scale=5.6,shrinkA=0,shrinkB=0,
                     color=c,lw=lw,linestyle=(0,(2.2,1.8)) if dashed else '-',zorder=3))

    def icon(name,x,y,size=8,color=INK,lw=.85):
        """Render the original 24-unit Lucide geometry as matplotlib vectors."""
        root=ET.parse(ICONS/f'{name}.svg').getroot()
        transform=Affine2D().scale(size/24).translate(x,y)+ax.transData
        for child in root:
            tag=child.tag.split('}')[-1]
            a=child.attrib
            if tag=='path':
                p=PathPatch(parse_path(a['d']),facecolor='none',edgecolor=color,
                            lw=lw,capstyle='round',joinstyle='round',transform=transform,zorder=6)
            elif tag=='circle':
                p=Circle((float(a['cx']),float(a['cy'])),float(a['r']),
                         facecolor='none',edgecolor=color,lw=lw,transform=transform,zorder=6)
            elif tag=='rect':
                p=FancyBboxPatch((float(a.get('x',0)),float(a.get('y',0))),
                      float(a['width']),float(a['height']),
                      boxstyle=f"round,pad=0,rounding_size={float(a.get('rx',0))}",
                      facecolor='none',edgecolor=color,lw=lw,transform=transform,zorder=6)
            else:
                raise ValueError(f'Unsupported SVG element {tag} in {name}')
            ax.add_patch(p)

    # a: illustrative task-clock durations, deliberately coarse, not measurements.
    # Cashier sequence: Education -> Business -> later fields of study.
    text(0,2,'a',9,bold=True)
    text(17,9,'Web retrieval task',8,bold=True,ha='center')
    text(17,13,'cashier task example',7,ha='center',color=SLATE)
    # Five equally spaced item centres; paired labels share a compact leading.
    item_centres=[22+11.5*i for i in range(5)]
    steps=[
        ('file-search','R1: Education','~10 min to answer',INK),
        ('clock','~45 min cooldown','research, prepare',SLATE),
        ('file-search','R2: Business','~1 min to answer',INK),
    ]
    for centre,(name,title,subtitle,color) in zip(item_centres,steps):
        icon(name,0.5,centre-3.5,7,color=color,lw=.8)
        text(9.5,centre-1.7,title,7.5,bold=True)
        text(9.5,centre+1.8,subtitle,7)
    for upper,lower in zip(item_centres[:-1],item_centres[1:]):
        middle=(upper+lower)/2
        arrow((4,middle-1),(4,middle+1),c=RULE,lw=.6)
    icon('repeat-2',0.5,item_centres[3]-3.5,7,color=SLATE,lw=.75)
    text(9.5,item_centres[3],'Repeat to R5',7)
    icon('square-x',0.5,item_centres[4]-3.5,7,color=SLATE,lw=.8)
    text(9.5,item_centres[4],'Sandbox\nshut down',7)

    # b: key directly above the agent; no panel title or unused top band.
    text(39,2,'b',9,bold=True)
    icon('eye',40,9.5,5,color=TEAL,lw=.75)
    arrow((46.5,12),(51.5,12),c=TEAL,dashed=True,lw=.8)
    text(53.5,12,'Read',7.5)
    icon('pencil',40,17,5,color=AMBER,lw=.75)
    arrow((46.5,19.5),(51.5,19.5),c=AMBER,lw=.8)
    text(53.5,19.5,'Write',7.5)

    box(40,29,24,25,fc='white',ec=RULE,lw=.65,r=1.5,linestyle=(0,(3,2)))
    icon('bot',45,30.5,14,color=INK,lw=1)
    text(52,49,'Agent',8,bold=True,ha='center')
    text(37.6,41.5,'Sandbox',7,ha='center',rotation=90,color=SLATE)
    text(52,62,'HTTP GET requests,\nallowed page edits',7,ha='center',color=SLATE)

    box(70,6,49,67,fc='white',ec=RULE,lw=.65,r=2)
    text(94.5,10.5,'Public wiki',8,bold=True,ha='center')
    rows=[
        ('list-ordered',21,'RecentChanges','New edits across the wiki,\nnewest at the top'),
        ('file-search',35.5,'Task pages','Questions and answers,\nretrieved by page name'),
        ('messages-square',50,'Notice boards','Links, access workarounds\nand coordination notes'),
        ('file-user',64.5,'Own page','Self-chosen agent name\nand current run status'),
    ]
    for name,y,title,description in rows:
        icon(name,72,y-4.6,8,color=AMBER if name=='messages-square' else TEAL,lw=.85)
        text(82.5,y-3.6,title,7.5,bold=True)
        text(82.5,y+1.6,description,7)

    # Ports preserve vertical order and avoid crossing read/write paths.
    arrow((70,20),(64,34),c=TEAL,rad=.12,dashed=True)
    arrow((70,33),(64,38),c=TEAL,rad=.05,dashed=True)
    arrow((64,41),(70,37.5),c=AMBER,rad=.05)
    arrow((70,48),(64,44),c=TEAL,rad=.05,dashed=True)
    arrow((64,47),(70,52),c=AMBER,rad=.05)
    arrow((64,50),(70,64.5),c=AMBER,rad=.12)

    # c: the record itself, restricted to the active weeks; no title, the bars take the height.
    text(NET['x0']-4,2,'c',9,bold=True)
    # Handles linked when they edited a task page in common, coloured by the task of their first task edit,
    # with area proportional to the number of links.
    net=json.loads((Path(CACHE)/'network.json').read_text())
    xy=np.array([[n['x'],n['y']] for n in net['nodes']])
    xy-=np.median(xy,0)
    r=np.linalg.norm(xy,axis=1,keepdims=True)
    xy=xy/np.maximum(r,1e-12)*r**.6            # radial compression: outlying chains do not shrink the core
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    xy=xy@vt.T                               # long axis horizontal
    lo,hi=xy.min(0),xy.max(0)
    scale=min((NET['x1']-NET['x0'])/(hi[0]-lo[0]),(NET['y1']-NET['y0'])/(hi[1]-lo[1]))
    xy=(xy-(lo+hi)/2)*scale+[(NET['x0']+NET['x1'])/2+NET['shift'],(NET['y0']+NET['y1'])/2]
    from matplotlib.collections import LineCollection
    task=[n['task'] for n in net['nodes']]
    top=[f for f,_ in collections.Counter(task).most_common(len(TASKS))]
    colour={f:c for f,c in zip(top,TASK_COLOURS)}
    ax.add_collection(LineCollection([xy[[u,v]] for u,v in net['edges']],colors=RULE,lw=.1,alpha=.35,zorder=1))
    degree=np.bincount(np.array(net['edges']).ravel(),minlength=len(task))
    # Area proportional to degree; coloured handles over grey ones, and within each the larger underneath.
    order=sorted(range(len(task)),key=lambda i: (task[i] in colour,-degree[i]))
    ax.scatter(xy[order,0],xy[order,1],s=.5+.07*degree[order],c=[colour.get(task[i],RULE) for i in order],
               lw=.15,edgecolors='white',zorder=3)
    for j,f in enumerate(top+['other']):
        x,y=NET['x0']+1.5+26*(j//6),NET['y1']+3+3.1*(j%6)
        ax.scatter([x],[y],s=9,c=colour.get(f,RULE),lw=0,zorder=3)
        text(x+2,y,TASKS.get(f,'other tasks'),7)
    OUT.mkdir(parents=True,exist_ok=True)
    assert all(7<=t.get_fontsize()<=9 for t in ax.texts)
    licence=(ICONS/'LICENSE').read_text()
    description='Icons: Lucide, https://lucide.dev. '+licence
    fig.savefig(OUT/'fig1a_schematic.pdf',facecolor='white',metadata={'Subject':description})
    fig.savefig(Path(PREVIEW)/'fig1a_schematic.png',dpi=300,facecolor='white',metadata={'Description':description})
    print('wrote paper/figs/fig1a_schematic.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
