def calibration_curve(title,Xaxis,Yaxis):
    """
    Makes a calibration curve of the given values.
    
    Input:
    - title : Has to be a string. Appears at the top of the chart and in the name of the saved file.
    - Xaxis : Has to be an array of the given concentration values.
    - Yaxis : Has to be an array of the given response values. 
    ! Xaxis and Yaxis MUST have the same length.
    
    Output: 
    slope, intercept, r_value, p_value, stderr : The parameters of the resulting equation.
    Chart depicting the resulting calibration curve in .png form.
    """
    fig = plt.figure(figsize =(5,5))
    fig.suptitle(title)
    ax = fig.add_subplot(1,1,1)
    ax2 = fig.add_subplot(1,1,1)

    linregress(Xaxis,Yaxis) #x and y are arrays or lists.
    slope, intercept, r_value, p_value, stderr = stats.linregress(Xaxis,Yaxis)

    mn=min(Xaxis)
    mx=max(Xaxis)
    x1=np.linspace(mn,mx)
    y1=slope*x1+intercept
    ax.plot(Xaxis,Yaxis,'ob')
    ax2.plot(x1,y1,'--r')
    ax.text(0.1*min(Xaxis), 0.9*max(Yaxis), 'y = ' + '{:.2f}'.format(intercept) + ' + {:.2f}'.format(slope) + 'x', size=14)
    ax.text(0.1*min(Xaxis), 0.8*max(Yaxis), '$R^{2}$=' + '{:.4f}'.format((r_value)**2), size=14)
    plt.savefig(title, bbox_inches = 'tight', dpi = 300) 
    return slope, intercept, r_value, p_value, stderr
