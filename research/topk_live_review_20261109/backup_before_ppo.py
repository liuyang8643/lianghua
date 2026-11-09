"""Create an explicit source/model checkpoint; never alter deployed defaults."""
import hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
from utils.atomic_file import atomic_write_json,file_sha256

ROOT=Path('checkpoints/pre_ppo_20261109')
def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    base=Path('strategy_release/new_model_topk_20261108');dest=Path('strategy_release/multistart_topk_20261109')
    assert not dest.exists();shutil.copytree(base,dest)
    manifest=json.loads((dest/'manifest.json').read_text(encoding='utf8'))
    winner=json.loads(Path('artifacts/research/topk_multistart_20261109/winner_lock.json').read_text())
    manifest.update(release_id='multistart_topk_20261109',display_name='冻结新模型＋多起点选中TopK',weights=[v/10 for v in winner['parameters']['weights']],parameters={k:v for k,v in winner['parameters'].items() if k!='weights'},selection_scope='2012-2021 chronological OOF annual scoring, nested12/48/all prior-year starts; locked before this outer evaluation. Not a new model fit.',winner_lock_sha256=file_sha256('artifacts/research/topk_multistart_20261109/winner_lock.json'))
    atomic_write_json(dest/'manifest.json',manifest)
    atomic_write_json('configs/selected_supervised_strategy_20261109.json',{'strategy':'new_model_topk','bundle':dest.as_posix(),'manifest_sha256':file_sha256(dest/'manifest.json'),'mvo_enabled':False})
    files=set()
    archives=[base/'research_source.zip',Path('artifacts/research/topk_multistart_20261109/source_snapshot.zip'),Path('artifacts/research/topk_multistart_20261109/environment/supplementary_dependency_sources.zip')]
    for archive in archives:
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                p=Path(name)
                if not name.startswith('.venv/') and p.is_file():files.add(p)
    for folder in Path('research').glob('topk_*_20261109'):
        files.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update([Path('configs/default_supervised_strategy.json'),Path('configs/selected_supervised_strategy_20261109.json'),Path('scripts/run_research_guarded.py')])
    sources={p.as_posix():file_sha256(p) for p in sorted(files)}
    archive=ROOT/'current_strategy_sources.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):z.write(p,p.as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in sources.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
    evidence={
       'winner_lock.json':'artifacts/research/topk_multistart_20261109/winner_lock.json',
       'selection_protocol.json':'research/topk_multistart_20261109/protocol.json',
       'stage48_protocol.json':'research/topk_multistart_refinement_20261109/protocol.json',
       'period_comparison.csv':'artifacts/research/topk_multistart_20261109/delivery/summary/period_comparison.csv',
       'paired_launch_comparison.csv':'artifacts/research/topk_launch_compare_20261109/report/paired_summary.csv',
       'paired_launch_distributions.json':'artifacts/research/topk_launch_compare_20261109/report/summary.json',
       'live_recommendation.json':'artifacts/research/topk_live_review_20261109/recommendation.json',
       'today_first_day.json':'artifacts/research/topk_today_20261109/result.json',
       'original_environment.json':'artifacts/research/topk_multistart_20261109/environment/manifest.json',
    }
    for name,path in evidence.items():shutil.copy2(path,ROOT/name)
    atomic_write_json(ROOT/'source_manifest.json',sources)
    atomic_write_json(ROOT/'checkpoint.json',{'state':'verified_code_and_strategy_backup','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'current_default_config_sha256':file_sha256('configs/default_supervised_strategy.json'),'selected_config_sha256':file_sha256('configs/selected_supervised_strategy_20261109.json'),'model_sha256':file_sha256(dest/'model.txt'),'sources_count':len(sources),'source_archive_sha256':file_sha256(archive),'source_manifest_sha256':file_sha256(ROOT/'source_manifest.json'),'evidence':{name:file_sha256(ROOT/name) for name in evidence},'scope':'All current strategy runtime/research dependencies from prior verified source closures plus this chat research modules. Both portable strategy bundles preserved. Unrelated dirty worktree changes are not directly staged. Raw market datasets/large intermediate arrays remain local and are not a remote data backup. No deployed default changed.'})
    print({'sources':len(sources),'archive_bytes':archive.stat().st_size,'selected_bundle':dest.as_posix()})

if __name__=='__main__':main()
