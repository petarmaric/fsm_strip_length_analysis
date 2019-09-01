import argparse
import logging
import os
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Agg') # Fixes weird segfaults, see http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server

from fsm_load_modal_composites import load_modal_composites
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


__version__ = '1.0.1'


FIGURE_SIZE = (11.7, 8.3) # In inches

DEFAULT_T_B = 6.35

SUBPLOTS_SPEC = [
    {
        'line_plots': [ # 1st key is the ``main_key``
            # key,           description
            ('omega',        'via direct method'),
            ('omega_approx', 'via physical dualism'),
        ],
        'ylabel': None, # if ``None``, automatically determine from ``main_key``
    },
    {
        'line_plots': [
            ('sigma_cr',        'via direct method'),
            ('sigma_cr_approx', 'via physical dualism'),
        ],
        'ylabel': None,
    },
    {
        'line_plots': [
            ('m_dominant', ''),
        ],
        'ylabel': 'dominant mode',
    },
    {
        'line_plots': [
            ('omega_rel_err',    'for natural frequency'),
            ('sigma_cr_rel_err', 'for critical buckling stress'),
        ],
        'ylabel': 'relative approximation errors',
    },
]


def plot_marker(x, y):
    plt.scatter(
        x,
        y,
        s=18,
        marker='o',
        label="%g [mm]" % x,
        zorder=100,
    )
    plt.annotate(
        "%g" % y,
        xy=(x, y),
        xytext=(0, +4),
        textcoords='offset points',
    )

def plot_markers(markers, x, y):
    if not markers:
        return

    mask = np.nonzero(np.in1d(x, markers, assume_unique=True))
    for marker_x, marker_y in zip(x[mask], y[mask]):
        plot_marker(marker_x, marker_y)

def find_automatic_markers_on_mode_transitions(modal_composites, x):
    data = modal_composites['m_dominant']
    idx = np.nonzero(np.ediff1d(data, to_begin=0))
    return x[idx]

def plot_modal_composite(modal_composites, column_units, column_descriptions, markers=None, add_automatic_markers=False):
    logging.info("Plotting modal composites...")
    start = timer()

    def _get_column_title(column_name):
        description = column_descriptions[column_name]
        unit = column_units[column_name]
        return description if not unit else "%s [%s]" % (description, unit)

    t_b = modal_composites['t_b'][0]
    plt.suptitle("modal composites, strip thickness %f [mm]" % t_b)

    x = modal_composites['a']
    min_x = np.min(x)
    max_x = np.max(x)

    markers = markers or []
    if add_automatic_markers:
        markers.extend(find_automatic_markers_on_mode_transitions(modal_composites, x))

    for ax_idx, spec in enumerate(SUBPLOTS_SPEC, start=1):
        plt.subplot(2, 2, ax_idx)

        main_key = spec['line_plots'][0][0] # 1st key is the ``main_key``
        for zorder, (key, description) in enumerate(spec['line_plots'], start=1):
            plt.plot(x, modal_composites[key], label=description, zorder=-zorder)

        plot_markers(markers, x, modal_composites[main_key])

        plt.xlim(min_x, max_x)

        plt.xlabel(_get_column_title('a'))
        plt.ylabel(spec['ylabel'] or _get_column_title(main_key))
        plt.legend()

    logging.info("Plotting completed in %f second(s)", timer() - start)

def dynamic_load_modal_composites(model_file, search_buffer=10**-10, **filters):
    modal_composites, column_units, column_descriptions = load_modal_composites(model_file, **filters)

    if modal_composites.size != 0:
        return modal_composites, column_units, column_descriptions

    t_b = filters.pop('t_b_fix')
    filters.update({
        't_b_min': t_b - search_buffer,
        't_b_max': t_b + search_buffer,
    })

    logging.warn("Could not find the exact value of t_b requested, expanding search condition to %(t_b_min)s <= t_b <= %(t_b_max)s", filters)
    return load_modal_composites(model_file, **filters)


def analyze_model(model_file, report_file, markers=None, add_automatic_markers=False, **filters):
    with PdfPages(report_file) as pdf:
        modal_composites, column_units, column_descriptions = dynamic_load_modal_composites(model_file, **filters)
        plot_modal_composite(modal_composites, column_units, column_descriptions, markers, add_automatic_markers)

        pdf.savefig()
        plt.close() # Prevents memory leaks

def configure_matplotlib():
    matplotlib.rc('figure',
        figsize=FIGURE_SIZE,
        titlesize='xx-large'
    )

    matplotlib.rc('figure.subplot',
        left   = 0.07, # the left side of the subplots of the figure
        right  = 0.98, # the right side of the subplots of the figure
        bottom = 0.06, # the bottom of the subplots of the figure
        top    = 0.91, # the top of the subplots of the figure
        wspace = 0.16, # the amount of width reserved for blank space between subplots
        hspace = 0.20, # the amount of height reserved for white space between subplots
    )

    matplotlib.rc('legend',
        fontsize='small',
    )

def main():
    # Setup command line option parser
    parser = argparse.ArgumentParser(
        description='Strip length-dependent visualization and modal analysis '\
                    'of the parametric model of buckling and free vibration '\
                    'in prismatic shell structures, as computed by the '\
                    'fsm_eigenvalue project.',
    )
    parser.add_argument(
        'model_file',
        help="File storing the computed parametric model"
    )
    parser.add_argument(
        '-r',
        '--report_file',
        metavar='FILENAME',
        help="Store the analysis report to the selected FILENAME, uses '<model_file>.pdf' by default"
    )
    parser.add_argument(
        '--a-min',
        metavar='VAL',
        type=float,
        help='If specified, clip the minimum strip length [mm] to VAL'
    )
    parser.add_argument(
        '--a-max',
        metavar='VAL',
        type=float,
        help='If specified, clip the maximum strip length [mm] to VAL'
    )
    parser.add_argument(
        '--t_b',
        metavar='VAL',
        type=float,
        default=DEFAULT_T_B,
        help="Plot figures by fixing the selected base strip thickness [mm] to VAL, %f by default" % DEFAULT_T_B
    )
    parser.add_argument(
        '--markers',
        metavar='POS',
        nargs='*',
        type=float,
        help='Plot marker(s) at specified strip length(s) [mm]'
    )
    parser.add_argument(
        '--add-automatic-markers',
        action='store_true',
        help='Plot automatic marker(s) on mode transitions'
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_const',
        const=logging.WARN,
        dest='verbosity',
        help='Be quiet, show only warnings and errors'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        const=logging.DEBUG,
        dest='verbosity',
        help='Be very verbose, show debug information'
    )
    parser.add_argument(
        '--version',
        action='version',
        version="%(prog)s " + __version__
    )
    args = parser.parse_args()

    # Configure logging
    log_level = args.verbosity or logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    configure_matplotlib()

    if not args.report_file:
        args.report_file = os.path.splitext(args.model_file)[0] + '.pdf'

    analyze_model(
        model_file=args.model_file,
        report_file=args.report_file,
        a_min=args.a_min,
        a_max=args.a_max,
        t_b_fix=args.t_b,
        markers=args.markers,
        add_automatic_markers=args.add_automatic_markers,
    )

if __name__ == '__main__':
    main()
