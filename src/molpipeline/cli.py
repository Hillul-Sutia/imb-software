"""Command-line interface for reproducible molecular experiments."""

import argparse
from pathlib import Path

from . import reporting
from ._artifacts import atomic_json, digest
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?')

    parser.add_argument(
        '--refresh-summaries', 
        metavar='DIRECTORY', 
        help='Refresh CSV summaries from saved results without training'
    )

    parser.add_argument(
        '--organize-results', 
        metavar='DIRECTORY', 
        help='Organize completed legacy results without retraining'
    )

    parser.add_argument(
        '--plan', action='store_true'
    )
    
    parser.add_argument(
        '--export-features', 
        action='store_true', 
        help='Export shared features and labels without splitting or training'
    )

    parser.add_argument(
        '--validate-only', 
        action='store_true', 
        help='Validate selected datasets without generating features or training'
    )
    
    args = parser.parse_args()
    
    if args.refresh_summaries:
        if args.config or args.plan or args.export_features or args.validate_only or args.organize_results:
            parser.error('--refresh-summaries must be used alone')

        directory = Path(args.refresh_summaries)
        
        if not directory.is_dir():
            parser.error('results directory does not exist')

        reporting.update_summaries(directory)
        print(f'Refreshed summaries in {directory.resolve()}')

        return
    
    if args.organize_results:
        if args.config or args.plan or args.export_features or args.validate_only:
            parser.error('--organize-results cannot be combined with config or --plan')

        organized, skipped = reporting.organize_existing(args.organize_results, atomic_json, digest)

        for message in organized + skipped:
            print(message)

        return
    
    if not args.config:
        parser.error('a config file is required')

    if sum((args.plan, args.export_features, args.validate_only)) > 1:
        parser.error('--plan, --export-features and --validate-only are mutually exclusive')
        
    run(args.config, args.plan, args.export_features, args.validate_only)
