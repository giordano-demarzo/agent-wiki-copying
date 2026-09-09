"""Figure 1: the task loop (a), what an agent reads and writes (b), and the record over time (c).

Panels a and b are vector illustration; panel c plots cache/timeline.json, which
src/timeline.py writes. All labels are editable text at 5 to 7 pt.
Icon source and licence: assets/icons/lucide/.
"""
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

from common import CACHE, FIGURES, ICONS as _ICONS
OUT=Path(FIGURES)
ICONS=Path(_ICONS)
# Exactly the palette of figures A-C: ink for the subject, teal for reading, amber for writing, slate and rule for
# everything passive. No tinted grounds beyond one neutral container, so the panels sit in the same visual world.
INK, SLATE, RULE='#1a1a1a','#6b7b8c','#b8bfc6'
TEAL, AMBER='#3d9aa1','#d99b3f'
GROUND='#f2f3f4'

# Three columns with 9 mm gutters: task loop 0-31, agent and wiki 40-120, timeline 136-178.
W_MM, H_MM=183, 77
TL=dict(x0=136, x1=178, ytop=13, ybot=62)


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
    box(0,6,31,67,fc=GROUND,ec='none',r=2)
    text(15.5,10,'Web Retrieval Task',5.8,bold=True,ha='center')
    text(15.5,13,'cashier task example',5.3,ha='center')
    # Five equally spaced item centres; paired labels share a compact leading.
    item_centres=[22+11*i for i in range(5)]
    steps=[
        ('file-search','R1: Education','~10 min to answer',INK),
        ('clock','~45 min cooldown','Research / prepare',SLATE),
        ('file-search','R2: Business','~1 min to answer',INK),
    ]
    for centre,(name,title,subtitle,color) in zip(item_centres,steps):
        icon(name,3,centre-3.5,7,color=color,lw=.75)
        text(12.5,centre-1.3,title,5.6,bold=True)
        text(12.5,centre+1.3,subtitle,5.3)
    for upper,lower in zip(item_centres[:-1],item_centres[1:]):
        middle=(upper+lower)/2
        arrow((6.5,middle-1),(6.5,middle+1),c=RULE,lw=.55)
    icon('repeat-2',3,item_centres[3]-3.5,7,color=SLATE,lw=.7)
    text(12.5,item_centres[3],'Repeat to R5',5.3)
    icon('square-x',3,item_centres[4]-3.5,7,color=SLATE,lw=.75)
    text(12.5,item_centres[4],'Sandbox\ndestroyed',5.5)

    # b: key directly above the agent; no panel title or unused top band.
    text(40,2,'b',9,bold=True)
    icon('eye',41,12,4.5,color=TEAL,lw=.7)
    arrow((47,14.25),(52,14.25),c=TEAL,dashed=True,lw=.75)
    text(54.5,14.25,'Read',5.6)
    icon('pencil',41,19,4.5,color=AMBER,lw=.7)
    arrow((47,21.25),(52,21.25),c=AMBER,lw=.75)
    text(54.5,21.25,'Write',5.6)

    box(40,29,26,25,fc='white',ec=RULE,lw=.65,r=1.5,linestyle=(0,(3,2)))
    icon('bot',45.5,31,15,color=INK,lw=1)
    text(53,49.7,'Agent',6.2,bold=True,ha='center')
    text(37.7,41.5,'Sandbox',5.5,ha='center',rotation=90)
    text(53,63,'HTTP GET requests\nallowed page edits',5.5,ha='center',color=SLATE)

    box(76,6,44,67,fc=GROUND,ec='none',r=2)
    text(98,10,'Public wiki',6.3,bold=True,ha='center')
    rows=[
        ('list-ordered',19,'RecentChanges','New edits across the wiki,\nlisted newest first.'),
        ('file-search',34,'Task pages','Questions and answers,\nretrieved by page name.'),
        ('messages-square',49,'Notice boards','Links, access workarounds\nand coordination notes.'),
        ('file-user',64,'Own page','Self-chosen agent name\nand current run status.'),
    ]
    for name,y,title,description in rows:
        icon(name,78.5,y-4.2,8.4,color=AMBER if name=='messages-square' else TEAL,lw=.8)
        text(89.5,y-3,title,6,bold=True)
        text(89.5,y+1.4,description,5.4)

    # Ports preserve vertical order and avoid crossing read/write paths.
    arrow((76,19),(66,34),c=TEAL,rad=.12,dashed=True)
    arrow((76,32),(66,38),c=TEAL,rad=.05,dashed=True)
    arrow((66,41),(76,36),c=AMBER,rad=.05)
    arrow((76,47),(66,44),c=TEAL,rad=.05,dashed=True)
    arrow((66,47),(76,51),c=AMBER,rad=.05)
    arrow((66,50),(76,64),c=AMBER,rad=.12)

    # c: the record itself, one bar per day, on the same axes as the rest of the paper.
    text(130,2,'c',9,bold=True)
    tl=json.loads((Path(CACHE)/'timeline.json').read_text())
    days=[datetime.date.fromisoformat(d) for d in tl['days']]
    axc=fig.add_axes([(TL['x0']+4.5)/W_MM,1-TL['ybot']/H_MM,(TL['x1']-TL['x0'])/W_MM,(TL['ybot']-TL['ytop'])/H_MM])
    if sum(tl['excluded']):   # only when the analysis filters a subpopulation out; empty when nothing is excluded
        axc.bar(days,np.array(tl['excluded'])+np.array(tl['edits']),color=RULE,width=0.8,label='excluded agents')
    axc.bar(days,tl['edits'],color=INK,width=0.8,label='edits')
    axc.bar(days,tl['new_handles'],color=AMBER,width=0.8,label='new handles')
    axc.set_yscale('symlog',linthresh=10); axc.set_ylim(0,15000)
    axc.set_ylabel('count per day',fontsize=6)
    MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']   # not %b: the system locale is not English
    axc.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v,p: f'{mdates.num2date(v).day:02d} {MON[mdates.num2date(v).month-1]}'))
    axc.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1,15]))
    axc.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[8,22]))
    axc.tick_params(labelsize=5.4,width=0.6,length=2.2)
    axc.legend(loc='upper left',frameon=False,fontsize=5,handlelength=1.1,handletextpad=0.5,borderpad=0,labelspacing=0.3)

    OUT.mkdir(parents=True,exist_ok=True)
    assert all(5<=t.get_fontsize()<=9 for t in ax.texts)
    licence=(ICONS/'LICENSE').read_text()
    description='Icons: Lucide, https://lucide.dev. '+licence
    for ext in ['pdf','svg','png']:
        metadata={'Subject':description} if ext=='pdf' else {'Description':description}
        fig.savefig(OUT/f'fig1a_schematic.{ext}',dpi=600,facecolor='white',metadata=metadata)
    plt.close(fig)


if __name__=='__main__':
    main()
