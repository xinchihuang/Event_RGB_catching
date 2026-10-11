"""Compare replay predictions to workbook IDs with explicit scene mapping."""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from project_paths import stage_path, resolve_path, setting_path
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np
import openpyxl

# Authoritative mapping supplied by the user. Do not use historical summary excel_id.
CASES = [('l', 'normal', 0), ('c', 'clutter', 100),
         ('dc', 'dark_clutter', 200), ('d', 'dark', 300)]
MODES = ['Model', 'RGB', 'Event']
COLORS = {'accurate': '#00877b', 'inaccurate': '#d64045', 'missing': '#728197'}


def read_ground_truth(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook['Sheet1']
    rows = iter(sheet.values)
    header = list(next(rows))
    columns = [header.index(k) for k in ['image_id', 'x_m', 'y_m']]
    result = {}
    for row in rows:
        if row[columns[0]] is None:
            continue
        number = float(row[columns[0]])
        if not number.is_integer() or not 0 <= number < 400:
            raise ValueError(f'Invalid workbook image_id: {number}')
        index = int(number)
        if index in result:
            raise ValueError(f'Duplicate workbook image_id: {index}')
        xy = np.array([row[columns[1]], row[columns[2]]], dtype=float)
        if not np.isfinite(xy).all():
            raise ValueError(f'Invalid ground truth coordinates for {index}')
        result[index] = xy
    workbook.close()
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(run, gt, offset, threshold):
    with (run / 'summary.csv').open() as stream:
        source = list(csv.DictReader(stream))
    lookup = {(r['mode'], r['task'], int(r['index'])): r for r in source}
    if len(lookup) != len(source):
        raise ValueError('Duplicate mode/task/index in summary')
    details, stats = [], []
    for mode in MODES:
        for case, task, base in CASES:
            if not any(key[:2] == (mode, task) for key in lookup):
                continue
            group = []
            for local in range(100):
                src = lookup.get((mode, task, local))
                if src is None:
                    raise ValueError(f'Missing summary row: {mode}/{task}/{local}')
                index = base + local
                raw = gt.get(index)
                truth = raw + offset if raw is not None else None
                row = dict(mode=mode, case=case, task=task, local_id=local,
                           excel_id=index, historical_excel_id=src['excel_id'],
                           run_status=src['status'], gt_x_m='', gt_y_m='',
                           aligned_gt_y_m='', aligned_gt_z_m='', pred_y_m='', pred_z_m='',
                           error_m='', outcome='', prediction_file='')
                if raw is not None:
                    row.update(gt_x_m=raw[0], gt_y_m=raw[1],
                               aligned_gt_y_m=truth[0], aligned_gt_z_m=truth[1])
                prediction = None
                if src['status'] == 'predicted':
                    path = run / 'results' / f'{mode}_{task}' / 'ball_pixel_3D_records' / f'{local}.csv'
                    data = np.loadtxt(path, delimiter=',')
                    if data.shape != (3,) or not np.isfinite(data).all() or np.allclose(data, 0):
                        raise ValueError(f'Invalid predicted result: {path}')
                    prediction = data[1:]
                    row.update(pred_y_m=prediction[0], pred_z_m=prediction[1], prediction_file=str(path))
                if truth is None:
                    row['outcome'] = 'missing_ground_truth'
                elif src['status'] != 'predicted':
                    row['outcome'] = src['status']
                else:
                    error = float(np.linalg.norm(prediction - truth))
                    row.update(error_m=error, outcome='accurate' if error <= threshold else 'inaccurate')
                group.append(row)
            errors = [r['error_m'] for r in group if r['error_m'] != '']
            accurate = sum(r['outcome'] == 'accurate' for r in group)
            eligible = sum(r['gt_x_m'] != '' and r['run_status'] != 'missing_input' for r in group)
            stats.append(dict(mode=mode, case=case, task=task, planned=len(group),
                              gt_available=sum(r['gt_x_m'] != '' for r in group),
                              missing_input=sum(r['run_status'] == 'missing_input' for r in group),
                              missing_gt=sum(r['gt_x_m'] == '' for r in group),
                              no_detection=sum(r['run_status'] == 'no_detection' for r in group),
                              timeout=sum(r['run_status'] == 'timeout' for r in group),
                              eligible=eligible, valid_prediction_pairs=len(errors), accurate=accurate,
                              inaccurate=len(errors)-accurate,
                              overall_success_pct=100 * accurate / eligible if eligible else '',
                              conditional_success_pct=100 * accurate / len(errors) if errors else '',
                              mean_error_m=float(np.mean(errors)) if errors else '',
                              median_error_m=float(np.median(errors)) if errors else '',
                              rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else ''))
            details.extend(group)
    return details, stats


def draw_panel(ax, rows, stat, limits, threshold):
    ax.set_facecolor('#fcfdff')
    ax.grid(alpha=.17, zorder=0)
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
    ax.set_aspect('equal', adjustable='box')
    valid_gt = [r for r in rows if r['gt_x_m'] != '']
    xy = np.array([[r['aligned_gt_y_m'], r['aligned_gt_z_m']] for r in valid_gt])
    ax.scatter(xy[:, 0], xy[:, 1], s=24, facecolors='none', edgecolors='#9ba6b4', linewidths=.8, zorder=2)
    segments = []
    for outcome in ['accurate', 'inaccurate']:
        selected = [r for r in rows if r['outcome'] == outcome]
        if selected:
            predictions = np.array([[r['pred_y_m'], r['pred_z_m']] for r in selected])
            ax.scatter(predictions[:, 0], predictions[:, 1], c=COLORS[outcome], s=28,
                       edgecolors='white', linewidths=.4, zorder=4, alpha=.88)
            segments.extend([[(r['aligned_gt_y_m'], r['aligned_gt_z_m']),
                              (r['pred_y_m'], r['pred_z_m'])] for r in selected])
    ax.add_collection(LineCollection(segments, colors='#8693a4', linewidths=.65, alpha=.55, zorder=1))
    failures = [r for r in valid_gt if r['outcome'] not in ('accurate', 'inaccurate')]
    if failures:
        points = np.array([[r['aligned_gt_y_m'], r['aligned_gt_z_m']] for r in failures])
        ax.scatter(points[:, 0], points[:, 1], marker='x', s=20, linewidths=.85,
                   c=COLORS['missing'], alpha=.65, zorder=3)
    ax.set_title(f"{stat['mode']} / {stat['case']}  |  error <= {threshold*100:g} cm: "
                 f"{stat['accurate']}/{stat['eligible']} ({stat['overall_success_pct']:.1f}%)", fontsize=10, pad=7)
    mean = f"{stat['mean_error_m']*100:.1f} cm" if stat['mean_error_m'] != '' else '—'
    ax.text(.025, .97, f"Valid predictions {stat['valid_prediction_pairs']}  ·  mean error {mean}\n"
            f"All-zero {stat['no_detection']}  ·  timeouts {stat['timeout']}  ·  missing inputs {stat['missing_input']}",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=3), zorder=5)
    ax.set_xlabel('Impact Y / m', fontsize=9); ax.set_ylabel('Impact Z / m', fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#c9d0da')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--xlsx', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--y-offset', type=float, default=-.05)
    parser.add_argument('--z-offset', type=float, default=.95)
    parser.add_argument('--threshold', type=float, default=.25)
    args = parser.parse_args()
    try:
        args.xlsx = setting_path('ground_truth_xlsx', args.xlsx, 'GROUND_TRUTH_XLSX')
        args.run_dir = resolve_path(args.run_dir)
        args.output_dir = resolve_path(args.output_dir)
        if not args.xlsx.is_file():
            raise FileNotFoundError(f'Missing workbook: {args.xlsx}')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.threshold <= 0: parser.error('threshold must be positive')
    gt = read_ground_truth(args.xlsx)
    rows, stats = compare(args.run_dir.resolve(), gt, np.array([args.y_offset,args.z_offset]), args.threshold)
    out=args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'sample_comparison.csv', rows)
    write_csv(out/'accuracy_summary.csv', stats)
    manifest = dict(workbook=str(args.xlsx.resolve()), run_dir=str(args.run_dir.resolve()),
                    mapping={case: [base,base+99] for case,_,base in CASES},
                    missing_workbook_ids=sorted(set(range(400))-set(gt)),
                    gt_transform=dict(Y='xlsx.x_m + y_offset', Z='xlsx.y_m + z_offset',
                                      y_offset=args.y_offset,z_offset=args.z_offset),
                    threshold_m=args.threshold,
                    success_denominator='Inputs with ground truth; zero predictions and timeouts count as failures; missing inputs/GT excluded.',
                    mean_error_denominator='Nonzero finite predictions with matched ground truth only.',
                    note='Workbook IDs use user-corrected l,c,dc,d mapping; historical summary excel_id ignored. Offsets inherited from existing plotting code, not refitted.')
    (out/'analysis_config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
    all_points = [v + [args.y_offset,args.z_offset] for v in gt.values()]
    all_points += [[r['pred_y_m'],r['pred_z_m']] for r in rows if r['pred_y_m'] != '']
    points=np.array(all_points)
    lo=np.floor((points.min(axis=0)-.12)*10)/10; hi=np.ceil((points.max(axis=0)+.12)*10)/10
    limits=[(lo[0],hi[0]),(lo[1],hi[1])]
    fig,axes=plt.subplots(3,4,figsize=(18,14))
    for ri,mode in enumerate(MODES):
        for ci,(case,task,base) in enumerate(CASES):
            ax=axes[ri,ci]
            stat=next((s for s in stats if s['mode']==mode and s['case']==case),None)
            if stat is None:
                ax.set_axis_off();ax.text(.5,.5,f'{mode} / {case}\nNot run in this batch',ha='center',va='center',color='#8b96a7',fontsize=14)
                continue
            group=[r for r in rows if r['mode']==mode and r['case']==case]
            draw_panel(ax,group,stat,limits,args.threshold)
            single,sax=plt.subplots(figsize=(7,7))
            draw_panel(sax,group,stat,limits,args.threshold)
            single.tight_layout();single.savefig(out/f'landing_{mode}_{case}.png',dpi=200);plt.close(single)
    fig.suptitle('Replay impact predictions vs. 400 measured throws',fontsize=22,y=.99)
    fig.text(.5,.952,'Matched by image_id: 0–99 l  |  100–199 c  |  200–299 dc  |  300–399 d',ha='center',fontsize=12)
    handles=[Line2D([],[],marker='o',ls='',markerfacecolor='none',markeredgecolor='#9ba6b4',label='Ground truth'),
             Line2D([],[],marker='o',ls='',color=COLORS['accurate'],label=f'Prediction error <= {args.threshold*100:g} cm'),
             Line2D([],[],marker='o',ls='',color=COLORS['inaccurate'],label=f'Prediction error > {args.threshold*100:g} cm'),
             Line2D([],[],marker='x',ls='',color=COLORS['missing'],label='No valid prediction (shown at ground truth)'),
             Line2D([],[],color='#8693a4',lw=.8,label='Ground truth to prediction for the same throw')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.941),ncol=5,frameon=False,fontsize=10)
    fig.text(.5,.04,f'Project coordinate conversion: Y = Excel x_m {args.y_offset:+g} m; Z = Excel y_m {args.z_offset:+g} m (not independently calibrated).',ha='center',fontsize=10)
    fig.text(.5,.023,'Success rate includes all-zero/timeouts; missing inputs/labels are excluded. Mean error uses valid predictions only. Missing detections are not plotted at the origin.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.055,right=.98,bottom=.08,top=.895,hspace=.34,wspace=.27)
    fig.savefig(out/'landing_accuracy_overview.png',dpi=200)
    fig.savefig(out/'landing_accuracy_overview.pdf')
    plt.close(fig)
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print('OUTPUT',out)


if __name__=='__main__':
    main()
