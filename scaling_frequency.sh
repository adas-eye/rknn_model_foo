
chip_name=0
freq_set_status=0

# order
# rk1808 0
# rk3399pro 1
# rv1109 2
# rv1126 3
# rk3566 4
# rk3568 5
# rk3588 6

### rv1103/rv1106 do not have bash command, not support now
# rv1103 7
# rv1106 8


usage()
{
    echo "USAGE: ./fixed_frequency.sh -c {chip_name} [-h]"
    echo "  -c:  chip_name, such as rv1126 / rk3588"
    echo "  -h:  Help"
}

# print_and_compare_result want result
print_and_compare_result()
{
    # echo "compare result"
    echo "    try set "$1
    echo "    query   "$2
    if [ $1 == $2 ];then
        echo "    Seting Success"
    else
        echo "    Seting Failed"
        freq_set_status=-1
    fi
}

# print_not_support_adjust {device} {freq want} {freq want}
print_not_support_adjust()
{
    echo "Firmware seems not support seting $1 frequency"
    echo "    wanted "$2
    echo "    query  "$3
}


# MAIN FUNCTION HERE #
vaild=0

if [ $# == 0 ]; then
    usage
    exit 0
fi

while getopts "c:dc" arg
do
    case $arg in
        c)
            chip_name=$OPTARG
        ;;
        h)
            usage
            exit 0
        ;;
        ?)
            usage
            exit 1
        ;;
    esac
done


if [ $chip_name == 'rv1126' ]; then
    seting_strategy=1
    CPU_freq=1512000
    NPU_freq=934000000
    DDR_freq=934000000
    elif [ $chip_name == 'rv1109' ]; then
    seting_strategy=1
    CPU_freq=1512000
    NPU_freq=594000000
    DDR_freq=934000000
    elif [ $chip_name == 'rk1808' ]; then
    seting_strategy=1
    # CPU_freq=1512000
    CPU_freq=1200000
    NPU_freq=792000000
    DDR_freq=934000000
    elif [ $chip_name == 'rk3399pro' ]; then
    seting_strategy=2
    CPU_freq=1512000
    NPU_freq=792000000
    DDR_freq=934000000
    elif [ $chip_name == 'rk3566' ]; then
    seting_strategy=3
    CPU_freq=1800000
    NPU_freq=1000000000
    DDR_freq=1056000000
    elif [ $chip_name == 'rk3568' ]; then
    seting_strategy=3
    CPU_freq=1800000
    NPU_freq=1000000000
    DDR_freq=1560000000
    elif [ $chip_name == 'rk3588' ]; then
    seting_strategy=4
    CPU_freq=2400000
    NPU_freq=1000000000
    DDR_freq=2112000000
    elif [ $chip_name == 'rv1106' ]; then
    seting_strategy=5
    CPU_freq=1608000
    NPU_freq=420000000
    DDR_freq=924000000
    elif [ $chip_name == 'rv1103' ]; then
    seting_strategy=5
    CPU_freq=1608000
    NPU_freq=420000000
    DDR_freq=924000000
else
    echo "$chip_name not recognize"
    exit 1
fi



echo "Try seting frequency for "${chip_name}
echo "    NPU seting to "$NPU_freq
echo "    CPU seting to "$CPU_freq
echo "    DDR seting to "$DDR_freq


case $seting_strategy in
    0)
        echo "seting strategy not implement now"
    ;;
    
    # for rk1808, rv1126, rv1109
    1)
        echo "CPU: seting frequency"
        echo userspace > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
        echo $CPU_freq > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
        cur_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq)
        print_and_compare_result $CPU_freq $cur_freq
        
        echo "NPU: seting frequency"
        echo userspace > /sys/class/devfreq/ffbc0000.npu/governor
        echo $NPU_freq > /sys/class/devfreq/ffbc0000.npu/userspace/set_freq
        cur_freq=$(cat /sys/class/devfreq/ffbc0000.npu/cur_freq)
        print_and_compare_result $NPU_freq $cur_freq
        
        print_not_support_adjust DDR $DDR_freq
        cat /sys/kernel/debug/clk/clk_summary | grep dpll
    ;;
    
    # for rk3399pro
    2)
        echo "seting strategy not implement now"
    ;;
    
    # for rk3566, rk3568
    3)
        echo "CPU: seting frequency"
        echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
        echo $CPU_freq > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
        cur_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq)
        print_and_compare_result $CPU_freq $cur_freq
        
        echo "NPU: seting frequency"
        echo userspace > /sys/class/devfreq/fde40000.npu/governor
        echo $NPU_freq > /sys/class/devfreq/fde40000.npu/userspace/set_freq
        cur_freq=$(cat /sys/devices/platform/fde40000.npu/devfreq/fde40000.npu/cur_freq)
        print_and_compare_result $NPU_freq $cur_freq
        
        echo "DDR: seting frequency"
        echo userspace > /sys/class/devfreq/dmc/governor
        echo $DDR_freq > /sys/class/devfreq/dmc/userspace/set_freq
        cur_freq=$(cat /sys/class/devfreq/dmc/cur_freq)
        print_and_compare_result $DDR_freq $cur_freq
    ;;
    
    # for rk3588
    4)
        echo "CPU: seting frequency"
        echo "  Core0"
        echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
        echo $CPU_freq > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
        cur_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq)
        print_and_compare_result $CPU_freq $cur_freq
        
        echo "  Core4"
        echo userspace > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
        echo $CPU_freq > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed
        cur_freq=$(cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_cur_freq)
        print_and_compare_result $CPU_freq $cur_freq
        
        echo "  Core6"
        echo userspace > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
        echo $CPU_freq > /sys/devices/system/cpu/cpufreq/policy6/scaling_setspeed
        cur_freq=$(cat /sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_cur_freq)
        print_and_compare_result $CPU_freq $cur_freq
        
        
        if [ -e  /sys/class/devfreq/fdab0000.npu/governor ];then
            echo "NPU: seting frequency"
            echo userspace > /sys/class/devfreq/fdab0000.npu/governor
            echo $NPU_freq > /sys/class/devfreq/fdab0000.npu/userspace/set_freq
            cur_freq=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq)
            print_and_compare_result $NPU_freq $cur_freq
        else
            cur_freq=$(cat /sys/kernel/debug/clk/scmi_clk_npu/clk_rate)
            print_not_support_adjust NPU $NPU_freq $cur_freq
        fi
        
        if [ -e /sys/class/devfreq/dmc/governor ];then
            echo "DDR: seting frequency"
            echo userspace > /sys/class/devfreq/dmc/governor
            echo $DDR_freq > /sys/class/devfreq/dmc/userspace/set_freq
            cur_freq=$(cat /sys/class/devfreq/dmc/cur_freq)
            print_and_compare_result $DDR_freq $cur_freq
        else
            print_not_support_adjust DDR $DDR_freq
            cat /sys/kernel/debug/clk/clk_summary | grep scmi_clk_ddr
        fi
        
    ;;
    
    5)
        echo "CPU: seting frequency"
        echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
        echo $CPU_freq > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
        cur_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq)
        print_and_compare_result $CPU_freq $cur_freq
        
        echo "no strategy to seting NPU frequency"
        
        echo "no strategy to seting DDR frequency"
    ;;
    *)
        echo "seting strategy not implement now"
    ;;
esac


echo $freq_set_status > ./freq_set_status