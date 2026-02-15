"""Main CLI module using Typer for command-line interface."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from place_recognition import __version__
from place_recognition.config import load_config
from place_recognition.data import DuplicateDetector, ManifestBuilder
from place_recognition.embeddings import EmbeddingExtractor
from place_recognition.evaluation import analyze_failures, evaluate_results
from place_recognition.index import SearchIndex
from place_recognition.logging_config import setup_logging
from place_recognition.utils.seeding import set_global_seed

app = typer.Typer(
    name="place-recognition",
    help="Image-Based Place Recognition & Retrieval System",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Display version and exit."""
    if value:
        console.print(f"[bold green]Place Recognition System[/bold green] v{__version__}")
        raise typer.Exit()


def common_setup(
    config_path: Optional[Path],
    seed: Optional[int],
    verbose: bool,
    device: Optional[str] = None,
) -> None:
    """Common setup for all CLI commands.
    
    Args:
        config_path: Path to config file
        seed: Random seed
        verbose: Enable verbose logging
        device: Device to use (cpu/cuda)
    """
    # Determine log level
    log_level = "DEBUG" if verbose else "INFO"
    
    # Setup logging
    setup_logging(level=log_level)
    
    # Load config
    overrides = {}
    if seed is not None:
        overrides["seed"] = seed
    if device is not None:
        overrides["device"] = device
    if verbose:
        overrides["log_level"] = "DEBUG"
    
    config = load_config(config_path, **overrides)
    
    # Set global seed
    set_global_seed(config.seed)
    
    return config


@app.command()
def extract_embeddings(
    manifest: Path = typer.Option(
        ...,
        "--manifest",
        "-m",
        help="Path to manifest file (gallery.jsonl or query.jsonl)",
        exists=True,
    ),
    dataset_root: Path = typer.Option(
        ...,
        "--dataset-root",
        "-d",
        help="Root directory of the dataset",
        exists=True,
    ),
    model: str = typer.Option(
        "resnet50",
        "--model",
        help="Pre-trained model name",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Batch size for inference",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device to use (cpu/cuda)",
    ),
    num_workers: int = typer.Option(
        0,
        "--num-workers",
        "-w",
        help="Number of parallel workers for image loading (0=sequential, 4=recommended)",
    ),
    cache_dir: Path = typer.Option(
        Path("embeddings_cache"),
        "--cache-dir",
        help="Directory to cache embeddings",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable caching",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)",
    ),
) -> None:
    """Extract embeddings from images using a pre-trained model.
    
    Processes images in batches and saves L2-normalized embeddings with caching.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    # Set seed if provided
    if seed is not None:
        set_global_seed(seed)
    
    console.print("[bold blue]Extract Embeddings[/bold blue]")
    console.print(f"Manifest: {manifest}")
    console.print(f"Model: {model}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Device: {device}")
    console.print(f"Num workers: {num_workers}")
    console.print(f"Cache: {'disabled' if no_cache else f'enabled ({cache_dir})'}")
    
    try:
        import time
        start_time = time.time()
        
        extractor = EmbeddingExtractor(
            model_name=model,
            device=device,
            batch_size=batch_size,
            cache_dir=cache_dir,
            num_workers=num_workers,
            verbose=verbose,
        )
        
        embeddings, records = extractor.extract_embeddings(
            manifest_path=manifest,
            dataset_root=dataset_root,
            use_cache=not no_cache,
        )
        
        elapsed_time = time.time() - start_time
        throughput = len(embeddings) / elapsed_time if elapsed_time > 0 else 0
        
        console.print(f"\n[green]SUCCESS:[/green] Extracted {len(embeddings)} embeddings")
        console.print(f"Feature dimension: {embeddings.shape[1]}")
        console.print(f"Shape: {embeddings.shape}")
        console.print(f"\n[bold]Performance:[/bold]")
        console.print(f"  Total time: {elapsed_time:.2f}s")
        console.print(f"  Throughput: {throughput:.1f} images/sec")
        console.print(f"  Avg time: {1000*elapsed_time/len(embeddings):.1f}ms/image")
        
    except Exception as e:
        console.print(f"\n[red]ERROR:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def build_index(
    embeddings_cache: Path = typer.Option(
        Path("embeddings_cache/gallery_resnet50_2db463d4.npz"),
        "--embeddings",
        "-e",
        help="Path to cached gallery embeddings file",
        exists=True,
    ),
    output: Path = typer.Option(
        Path("gallery_index.npz"),
        "--output",
        "-o",
        help="Output path for search index",
    ),
    use_faiss: bool = typer.Option(
        False,
        "--use-faiss",
        help="Use FAISS for indexing (faster for large datasets)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)",
    ),
) -> None:
    """Build search index from extracted gallery embeddings.
    
    Loads gallery embeddings and builds an efficient search index
    for fast similarity-based retrieval.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    console.print("[bold blue]Build Index[/bold blue]")
    console.print(f"Embeddings: {embeddings_cache}")
    console.print(f"Output: {output}")
    console.print(f"Use FAISS: {use_faiss}")
    
    try:
        # Load embeddings
        console.print("\nLoading embeddings...")
        import numpy as np
        data = np.load(embeddings_cache, allow_pickle=True)
        embeddings = data["embeddings"]
        records = data["records"].tolist()
        
        console.print(f"Loaded {len(embeddings)} gallery embeddings")
        
        # Build index
        index = SearchIndex(
            embeddings=embeddings,
            records=records,
            index_type="flat",
            use_faiss=use_faiss,
        )
        index.build()
        
        # Save index
        index.save(output)
        
        console.print(f"\n[green]SUCCESS:[/green] Index built successfully")
        console.print(f"Gallery size: {len(embeddings)}")
        console.print(f"Feature dimension: {embeddings.shape[1]}")
        console.print(f"Index saved to: {output}")
    
    except Exception as e:
        console.print(f"\n[red]ERROR:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def search(
    gallery_embeddings: Path = typer.Option(
        ...,
        "--gallery",
        "-g",
        help="Gallery embeddings (.npz file)",
        exists=True,
    ),
    query_embeddings: Path = typer.Option(
        ...,
        "--query",
        "-q",
        help="Query embeddings (.npz file)",
        exists=True,
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Number of top results to retrieve",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for search results (JSON)",
    ),
    use_faiss: bool = typer.Option(
        False,
        "--use-faiss",
        help="Use FAISS for faster search",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)",
    ),
) -> None:
    """Search for similar places using query images.
    
    Performs similarity search between query and gallery embeddings,
    returning Top-K most similar places for each query.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    console.print("[bold blue]Search for Similar Places[/bold blue]")
    console.print(f"Gallery: {gallery_embeddings}")
    console.print(f"Query: {query_embeddings}")
    console.print(f"Top-K: {top_k}")
    console.print(f"Use FAISS: {use_faiss}")
    
    try:
        import json
        import numpy as np
        from place_recognition.index import SearchIndex
        
        # Load embeddings
        console.print("\n[1/3] Loading embeddings...")
        gallery_data = np.load(gallery_embeddings, allow_pickle=True)
        query_data = np.load(query_embeddings, allow_pickle=True)
        
        gallery_embs = gallery_data["embeddings"]
        gallery_recs = gallery_data["records"]
        query_embs = query_data["embeddings"]
        query_recs = query_data["records"]
        
        console.print(f"  Gallery: {len(gallery_embs)} embeddings")
        console.print(f"  Query: {len(query_embs)} embeddings")
        
        # Build index
        console.print("\n[2/3] Building search index...")
        index = SearchIndex(
            embeddings=gallery_embs,
            records=gallery_recs,
            use_faiss=use_faiss,
        )
        index.build()
        
        # Search
        console.print("\n[3/3] Searching...")
        results = index.search(query_embs, top_k=top_k)
        
        # Display sample results
        console.print("\n[bold]Sample Results (first 3 queries):[/bold]")
        for i, query_results in enumerate(results[:3]):
            query_place = query_recs[i]["place_id"]
            query_path = query_recs[i]["image_path"]
            
            console.print(f"\n  Query {i+1}: {query_place}")
            console.print(f"    Image: {query_path}")
            console.print(f"    Top-{min(3, len(query_results))} matches:")
            
            for j, result in enumerate(query_results[:3], 1):
                console.print(
                    f"      {j}. {result['place_id']} "
                    f"(similarity: {result['similarity']:.4f})"
                )
        
        # Save results if requested
        if output:
            console.print(f"\n[bold]Saving results to:[/bold] {output}")
            output_data = {
                "gallery_embeddings": str(gallery_embeddings),
                "query_embeddings": str(query_embeddings),
                "top_k": top_k,
                "use_faiss": use_faiss,
                "results": [
                    {
                        "query_idx": i,
                        "query_place_id": query_recs[i]["place_id"],
                        "query_image_path": query_recs[i]["image_path"],
                        "top_k_results": query_results[:top_k],
                    }
                    for i, query_results in enumerate(results)
                ],
            }
            
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
        
        console.print(f"\n[green]SUCCESS:[/green] Search complete")
        console.print(f"  Total queries: {len(results)}")
        console.print(f"  Top-K per query: {top_k}")
        
        if not output:
            console.print("\n[yellow]TIP:[/yellow] Use --output to save results to JSON")
        
    except Exception as e:
        console.print(f"\n[red]ERROR:[/red] Search failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    gallery_embeddings: Path = typer.Option(
        ...,
        "--gallery-embeddings",
        "-g",
        help="Path to gallery embeddings cache file",
        exists=True,
    ),
    query_embeddings: Path = typer.Option(
        ...,
        "--query-embeddings",
        "-q",
        help="Path to query embeddings cache file",
        exists=True,
    ),
    top_k: int = typer.Option(
        20,
        "--top-k",
        "-k",
        help="Number of top results to retrieve for evaluation",
    ),
    recall_ks: str = typer.Option(
        "1,5,10,20",
        "--recall-ks",
        help="Comma-separated K values for Recall@K (e.g., '1,5,10')",
    ),
    failure_top_k: int = typer.Option(
        5,
        "--failure-top-k",
        help="Top-K to use for failure analysis (default: 5)",
    ),
    confidence_strategy: str = typer.Option(
        "none",
        "--confidence-strategy",
        help="Open-set strategy: 'none', 'max_similarity', 'margin'",
    ),
    unknown_threshold: float = typer.Option(
        0.5,
        "--unknown-threshold",
        help="Threshold for UNKNOWN prediction",
    ),
    use_faiss: bool = typer.Option(
        False,
        "--use-faiss",
        help="Use FAISS for faster search",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for evaluation report (JSON)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)",
    ),
) -> None:
    """Evaluate retrieval performance on query set.
    
    This command:
    1. Loads gallery and query embeddings
    2. Builds search index
    3. Performs retrieval for all queries
    4. Computes evaluation metrics (Recall@K, mAP)
    5. Optionally saves detailed report
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    console.print("[bold blue]Evaluate Retrieval Performance[/bold blue]")
    console.print(f"Gallery embeddings: {gallery_embeddings}")
    console.print(f"Query embeddings: {query_embeddings}")
    console.print(f"Top-K: {top_k}")
    console.print(f"Recall@K values: {recall_ks}")
    console.print(f"Confidence strategy: {confidence_strategy}")
    if confidence_strategy != "none":
        console.print(f"Unknown threshold: {unknown_threshold}")
    console.print(f"Use FAISS: {use_faiss}")
    
    try:
        import numpy as np
        from place_recognition.evaluation import (
            analyze_failure_patterns,
            analyze_failures,
            apply_open_set_strategy,
            evaluate_open_set_impact,
            evaluate_results,
            print_failure_analysis,
        )
        
        # Parse and validate recall K values
        try:
            ks = [int(k.strip()) for k in recall_ks.split(",") if k.strip()]
            ks = sorted(set(k for k in ks if k > 0))  # Remove duplicates, sort, filter invalid
            if not ks:
                ks = [1, 5, 10, 20]  # Fallback to default
                console.print("[yellow]Warning:[/yellow] Invalid recall_ks, using default: [1, 5, 10, 20]")
        except ValueError as e:
            console.print(f"[red]ERROR:[/red] Invalid recall_ks format: {recall_ks}")
            console.print("Expected format: '1,5,10,20'")
            raise typer.Exit(code=1)
        
        console.print(f"Evaluating Recall@K for K = {ks}")
        
        # Load gallery embeddings
        console.print("\n[1/5] Loading gallery embeddings...")
        gallery_data = np.load(gallery_embeddings, allow_pickle=True)
        gallery_embs = gallery_data["embeddings"]
        gallery_recs = gallery_data["records"].tolist()
        console.print(f"  Gallery: {len(gallery_embs)} images")
        
        # Load query embeddings
        console.print("\n[2/5] Loading query embeddings...")
        query_data = np.load(query_embeddings, allow_pickle=True)
        query_embs = query_data["embeddings"]
        query_recs = query_data["records"].tolist()
        console.print(f"  Queries: {len(query_embs)} images")
        
        # Build search index
        console.print("\n[3/5] Building search index...")
        index = SearchIndex(
            embeddings=gallery_embs,
            records=gallery_recs,
            index_type="flat",
            use_faiss=use_faiss,
        )
        index.build()
        console.print(f"  Index built in {index.stats['build_time']:.3f}s")
        
        # Perform search
        console.print(f"\n[4/5] Searching {len(query_embs)} queries...")
        results = index.search(query_embs, top_k=top_k)
        console.print(f"  Search completed")
        
        # Apply open-set strategy
        open_set_impact = None
        if confidence_strategy != "none":
            console.print(f"\n[4.5/5] Applying open-set strategy ({confidence_strategy}, threshold={unknown_threshold})...")
            original_results = results
            results = apply_open_set_strategy(results, confidence_strategy, unknown_threshold)
            
            # Compute impact
            open_set_impact = evaluate_open_set_impact(original_results, results, query_recs)
            console.print(f"  UNKNOWN predictions: {open_set_impact['unknown_predictions']}/{open_set_impact['total_queries']} ({open_set_impact['unknown_rate']:.2%})")
            console.print(f"  False positives removed: {open_set_impact['false_positives_removed']}")
            console.print(f"  True positives removed: {open_set_impact['true_positives_removed']}")
        
        # Compute metrics
        console.print("\n[5/5] Computing evaluation metrics...")
        metrics = evaluate_results(results, query_recs, ks=ks)
        
        # Display results
        console.print("\n" + "="*70)
        console.print("[bold green]EVALUATION RESULTS[/bold green]")
        console.print("="*70)
        
        console.print(f"\n[bold]Dataset:[/bold]")
        console.print(f"  Gallery images: {len(gallery_embs):,}")
        console.print(f"  Query images:   {len(query_embs):,}")
        
        console.print(f"\n[bold]Metrics:[/bold]")
        for k in ks:
            recall_val = metrics[f"recall@{k}"]
            color = "green" if recall_val > 0.8 else "yellow" if recall_val > 0.5 else "red"
            console.print(f"  Recall@{k:<3}: [{color}]{recall_val:.2%}[/{color}]")
        
        map_val = metrics["mAP"]
        color = "green" if map_val > 0.8 else "yellow" if map_val > 0.5 else "red"
        console.print(f"  mAP:        [{color}]{map_val:.2%}[/{color}]")
        
        console.print(f"\n[bold]Performance:[/bold]")
        if "avg_search_time" in index.get_stats():
            avg_time = index.get_stats()["avg_search_time"] * 1000
            console.print(f"  Avg search time: {avg_time:.2f}ms/query")
        
        # Failure analysis
        console.print(f"\n[bold]Failure Analysis:[/bold]")
        failure_analysis = analyze_failures(results, query_recs, top_k=failure_top_k)
        console.print(f"  Failed queries: {failure_analysis['num_failures']}/{len(query_recs)} (top-{failure_top_k})")
        console.print(f"  Failure rate:   {failure_analysis['failure_rate']:.2%}")
        
        if failure_analysis['num_failures'] > 0:
            console.print(f"\n  Top-3 Failures:")
            for i, failure in enumerate(failure_analysis['failures'][:3], 1):
                console.print(f"    {i}. Query: {failure['query_place_id']}")
                console.print(f"       Top-1 retrieved: {failure['top1_place_id']} "
                            f"(similarity: {failure['top1_similarity']:.4f})")
            
            # Pattern analysis
            console.print(f"\n[bold]Failure Patterns:[/bold]")
            pattern_analysis = analyze_failure_patterns(results, query_recs, top_k=failure_top_k)
            
            console.print(f"\n  Top Confusion Pairs:")
            for i, ((query, retrieved), count) in enumerate(pattern_analysis['confusion_pairs'][:3], 1):
                console.print(f"    {i}. {query} → {retrieved} ({count}x)")
            
            margin = pattern_analysis['margin_stats']
            if margin['count'] > 0:
                console.print(f"\n  Similarity Margins:")
                console.print(f"    Mean: {margin['mean']:.4f} (top-1 vs correct)")
        
        # Save report if requested
        if output:
            console.print(f"\n[bold]Saving report...[/bold]")
            import json
            
            # Get performance stats
            index_stats = index.get_stats()
            
            report = {
                "gallery_embeddings": str(gallery_embeddings),
                "query_embeddings": str(query_embeddings),
                "num_gallery": len(gallery_embs),
                "num_queries": len(query_embs),
                "feature_dim": gallery_embs.shape[1],
                "top_k": top_k,
                "index_config": {
                    "index_type": "flat",
                    "use_faiss": use_faiss,
                },
                "open_set_config": {
                    "strategy": confidence_strategy,
                    "threshold": unknown_threshold,
                },
                "metrics": metrics,
                "performance": {
                    "index_build_time_s": index_stats['build_time'],
                    "total_search_time_s": index_stats['total_search_time'],
                    "avg_search_time_ms": index_stats.get('avg_search_time', 0) * 1000,
                    "total_queries": index_stats['total_queries'],
                },
                "failure_analysis": {
                    "failure_top_k": failure_top_k,
                    "num_failures": failure_analysis['num_failures'],
                    "failure_rate": failure_analysis['failure_rate'],
                    "failures": failure_analysis['failures'][:10],  # Top 10 failures
                    "patterns": analyze_failure_patterns(results, query_recs, top_k=failure_top_k) if failure_analysis['num_failures'] > 0 else {},
                },
            }
            
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            console.print(f"  Report saved to: {output}")
        
        console.print("\n" + "="*70)
        console.print(f"[green]✓ Evaluation complete![/green]")
        
    except Exception as e:
        console.print(f"\n[red]ERROR:[/red] {str(e)}")
        import traceback
        if verbose:
            traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def build_manifest(
    dataset_root: Path = typer.Option(
        ...,
        "--dataset-root",
        "-d",
        help="Root directory containing place folders",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("manifests"),
        "--output-dir",
        "-o",
        help="Directory to save manifest files",
    ),
    min_resolution: int = typer.Option(
        32,
        "--min-resolution",
        "-m",
        help="Minimum width/height threshold",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)",
    ),
) -> None:
    """Build dataset manifests by scanning place folders.
    
    Scans dataset_root for place directories containing gallery/ and query/
    subdirectories. Creates JSONL manifest files with validated images.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    console.print("[bold blue]Build Manifest[/bold blue]")
    console.print(f"Dataset root: {dataset_root}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Min resolution: {min_resolution}x{min_resolution}")
    
    try:
        builder = ManifestBuilder(
            dataset_root=dataset_root,
            output_dir=output_dir,
            min_resolution=min_resolution,
            verbose=verbose,
        )
        builder.build()
        console.print("\n[green]SUCCESS:[/green] Manifest built successfully")
        console.print(f"Check {output_dir} for gallery.jsonl, query.jsonl, and stats.json")
    except Exception as e:
        console.print(f"\n[red]ERROR:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def check_duplicates(
    manifest_dir: Path = typer.Option(
        Path("manifests"),
        "--manifest-dir",
        "-m",
        help="Directory containing manifest files",
        exists=True,
    ),
    dataset_root: Path = typer.Option(
        ...,
        "--dataset-root",
        "-d",
        help="Root directory of the dataset",
        exists=True,
    ),
    hamming_threshold: int = typer.Option(
        5,
        "--hamming-threshold",
        "-t",
        help="Maximum Hamming distance for near-duplicates",
    ),
    output: Path = typer.Option(
        Path("duplicate_report.json"),
        "--output",
        "-o",
        help="Output path for report",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)",
    ),
) -> None:
    """Check for duplicate and near-duplicate images across splits.
    
    This is critical for preventing data leakage between gallery and query sets.
    Uses MD5 for exact duplicates and perceptual hashing for near-duplicates.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)
    
    console.print("[bold blue]Check Duplicates[/bold blue]")
    console.print(f"Manifest directory: {manifest_dir}")
    console.print(f"Dataset root: {dataset_root}")
    console.print(f"Hamming threshold: {hamming_threshold}")
    
    try:
        detector = DuplicateDetector(
            manifest_dir=manifest_dir,
            dataset_root=dataset_root,
            hamming_threshold=hamming_threshold,
            verbose=verbose,
        )
        no_leakage = detector.run(output)
        
        if no_leakage:
            console.print("\n[green]SUCCESS:[/green] No critical data leakage detected")
        else:
            console.print("\n[red]CRITICAL:[/red] Data leakage detected - see report")
            raise typer.Exit(code=1)
        
        console.print(f"Report saved to: {output}")
    
    except Exception as e:
        console.print(f"\n[red]ERROR:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Place Recognition & Retrieval System - Main CLI."""
    pass


if __name__ == "__main__":
    app()
