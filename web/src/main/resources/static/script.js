// ==========================================
// 1. CẤU HÌNH & TIỆN ÍCH DÙNG CHUNG
// ==========================================
const API = "http://localhost:8080";
const ENDPOINTS = { LOCKERS: `${API}/lockers`, SESSIONS: `${API}/sessions`, TICKETS: `${API}/tickets` };
let chartInstance = null;
let barChartInstance = null;
let trendChartInstance = null;
let isTrendChartInitialized = false;
let isSearching = false;
let originalTicketsData = [];
// Hàm Generic gọi API tập trung, tự động bắt lỗi hệ thống
async function apiCall(url, options = {}) {
    try {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error(`Hệ thống báo lỗi: ${res.status}`);
        return res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
    } catch (e) {
        showErrorDialog(e.message);
    }
}


function formatSessionFolderId(rawTime) {
    if (!rawTime) return "";
    const d = new Date(rawTime);

    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    const hh = String(d.getHours()).padStart(2, '0');
    const min = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');

    return `${yyyy}${mm}${dd}_${hh}${min}${ss}`;
}

// Trả về class CSS theo trạng thái tủ đồ
const getStatusClass = (s = "") => {
    if (!s) return "error";
    const statusLower = s.trim().toLowerCase();
    if (statusLower === "available" || statusLower === "free") return "available";
    if (statusLower === "occupied" || statusLower === "in-use") return "occupied";
    return "error";
};
// Khởi chạy
document.addEventListener("DOMContentLoaded", () => {
    loadPageData();
    setInterval(() => {
        if (!document.querySelector('.modal[style*="display: block"]') && !isSearching) loadPageData();
    }, 15000);
});

function loadPageData() {
    const path = window.location.pathname;
    if (path === "/" || path.endsWith("index.html") || path === "") loadDashboard();
    else if (path.includes("lockers.html")) loadLockers();
    else if (path.includes("users.html")) loadSessions();
    else if (path.includes("support.html")) loadTickets();
}

function showErrorDialog(msg) {
    const dialog = document.getElementById("errorDialog");
    if (dialog) { document.getElementById("errorDialogMessage").innerText = msg; dialog.style.display = "flex"; }
    else alert(msg);
}

function closeErrorDialog() { document.getElementById("errorDialog").style.display = "none"; }
function showSuccessDialog(message) {
    document.getElementById("successDialogMessage").textContent = message;
    document.getElementById("successDialog").style.display = "flex";
}

function closeSuccessDialog() {
    document.getElementById("successDialog").style.display = "none";
}
function closeModal() { ['lockerModal', 'error-message'].forEach(id => { const el = document.getElementById(id); if (el) el.style.display = "none"; }); }
function closeTicketModal() { document.getElementById("ticketModal").style.display = "none"; }

// ==========================================
// 2. QUẢN LÝ LOCKER & KHẨN CẤP (lockers.html)
// ==========================================
function renderTable(data) {
    const tbody = document.getElementById("lockerTable");
    if (!tbody) return;
    tbody.innerHTML = data?.map(l => `
        <tr>
            <td><strong>${l.id}</strong></td>
            <td>${l.location || 'Chưa xác định'}</td>
            <td class="status-${getStatusClass(l.status)}">${l.status}</td>
            <td>
${l.status?.toUpperCase() === "OCCUPIED" ? `<button style="background:#3b82f6" data-location="${l.location || 'Chưa xác định'}" onclick="handleOpenTicket('${l.id}', this)">Mở</button>` : ''}                <button style="background:#475569" onclick="showEditLocker('${l.id}')">Sửa</button>
                <button style="background:#ef4444" onclick="deleteLocker('${l.id}')">Xóa</button>
            </td>
        </tr>
    `).join("") || `<tr><td colspan="4" style="text-align:center;">Không tìm thấy dữ liệu phù hợp.</td></tr>`;
}

const loadLockers = async () => renderTable(await apiCall(ENDPOINTS.LOCKERS));

async function filterLockers() {
    const keyword = document.getElementById("lockerSearch").value.trim();
    const status = document.getElementById("statusFilter").value;
    isSearching = (keyword !== "" || status !== "ALL");
    renderTable(await apiCall(`${ENDPOINTS.LOCKERS}/search?keyword=${encodeURIComponent(keyword)}&status=${status}`));
}

function resetFilters() { document.getElementById("lockerSearch").value = ""; document.getElementById("statusFilter").value = "ALL"; loadLockers(); }

async function handleOpenTicket(lockerId, buttonElement) {
    const lockerLocation = buttonElement ? buttonElement.getAttribute("data-location") : "Chưa xác định";

    const session = await apiCall(`${ENDPOINTS.SESSIONS}/${lockerId}/current-session`);

    if (!session) return alert("⚠️ Không tìm thấy phiên sử dụng hoạt động.");
    if (!session.startTime && !session.start_time) return alert("❌ Phiên không có thời gian bắt đầu!");

    showTicketModal(lockerId, lockerLocation, formatSessionFolderId(session.startTime || session.start_time));
}
async function showTicketModal(lockerId, lockerLocation, sessionId) {
    document.getElementById("ticketModal").style.display = "block";

    document.getElementById("displayLockerId").innerText = ` - Vị trí: ${lockerLocation}`;

    document.getElementById("ticketLockerId").value = lockerId;
    document.getElementById("ticketTime").value = new Date().toLocaleString('vi-VN');

    const gallery = document.getElementById("imageGallery"), placeholder = document.getElementById("imagePlaceholder");
    gallery.querySelectorAll('img').forEach(img => img.remove());
    placeholder.style.display = "block";

    const imageUrls = await apiCall(`${API}/sessions/images-by-session?sessionId=${sessionId}`);

    if (imageUrls?.length > 0) {
        placeholder.style.display = "none";
        imageUrls.forEach(url => {
            const img = document.createElement("img");
            img.src = `${API}${url}`;
            img.style = "width: 100%; height: 120px; object-fit: cover; border-radius: 8px; cursor: pointer;";
            img.onclick = () => window.open(img.src, '_blank');
            gallery.appendChild(img);
        });
    } else {
        placeholder.innerHTML = "⚠️ Không tìm thấy dữ liệu ảnh khẩn cấp.";
    }
}
async function submitTicket() {
    const lockerId = document.getElementById("ticketLockerId").value;
    const reason = document.getElementById("ticketReason").value.trim();

    if (!reason) {
        showErrorDialog("Vui lòng nhập lý do mở tủ!");
        return;
    }

    // Sử dụng fetch trực tiếp hoặc qua cấu trúc kiểm soát của bạn để kích hoạt lệnh điều khiển rơ-le
    try {
        const response = await fetch(`${API}/lockers/open`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lockerId: lockerId, reason: reason })
        });

        if (response.ok) {
            closeTicketModal();
            showSuccessDialog(
                "Đã gửi lệnh mở tủ thực tế và lưu Ticket thành công!"
            );
            loadLockers(); // Tải lại danh sách tủ để cập nhật trạng thái mới lên màn hình
        } else {
            showErrorDialog(`Không thể mở tủ, Lỗi kết nối hoặc thiết bị gặp sự cố!`);
        }
    } catch (e) {
        showErrorDialog("Lỗi kết nối Server phần cứng!");
    }
}
async function showEditLocker(id) {
    const l = await apiCall(`${ENDPOINTS.LOCKERS}/${id}`);
    if (!l) return;
    document.getElementById("modalTitle").innerText = "Cập Nhật Tủ #" + id;
    document.getElementById("editFlag").value = "EDIT";
    document.getElementById("idInputGroup").style.display = "block";
    const idInput = document.getElementById("lockerId"); idInput.value = id; idInput.disabled = true;
    document.getElementById("lockerLocation").value = l.location || "";
    document.getElementById("lockerStatus").value = l.status;
    document.getElementById("statusGroup").style.display = "block";
    document.getElementById("lockerModal").style.display = "block";
}

function showLockerModal() {
    document.getElementById("modalTitle").innerText = "Thêm Tủ Đồ Mới";
    document.getElementById("editFlag").value = "";
    document.getElementById("idInputGroup").style.display = "none";
    document.getElementById("lockerLocation").value = "";
    document.getElementById("statusGroup").style.display = "none";
    document.getElementById("lockerModal").style.display = "block";
}

async function saveLocker() {
    const loc = document.getElementById("lockerLocation").value.trim();
    if (!loc) { document.getElementById("error-message").innerText = "⚠️ Chưa nhập vị trí!"; return; }
    const isEdit = document.getElementById("editFlag").value === "EDIT";
    const res = await fetch(isEdit ? `${ENDPOINTS.LOCKERS}/${document.getElementById("lockerId").value}` : ENDPOINTS.LOCKERS, {
        method: isEdit ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(isEdit ? { location: loc, status: document.getElementById("lockerStatus").value } : { location: loc })
    });
    if (res.ok) { closeModal(); loadLockers(); } else showErrorDialog("Vị trí này đã có tủ đồ rồi!");
}

async function deleteLocker(id) { if (confirm(`Xóa tủ ${id}?`)) { await fetch(`${ENDPOINTS.LOCKERS}/${id}`, { method: "DELETE" }); loadLockers(); } }

// ==========================================
// 3. DASHBOARD & TICKETS (index.html / support.html)
// ==========================================
async function loadDashboard() {
    const stats = await apiCall(`${API}/lockers/dashboard/stats`);
    if (!stats) return;
    console.log("DỮ LIỆU STATS THỰC TẾ:", stats);

    // 1. Đổ dữ liệu lên các thẻ Card thống kê (Đã sửa lại id="available" theo đúng HTML của Toàn)
    if (document.getElementById("total")) document.getElementById("total").innerText = stats.total || 0;
    if (document.getElementById("available")) document.getElementById("available").innerText = stats.free || 0; // 🔥 Đã khớp id="available"
    if (document.getElementById("occupied")) document.getElementById("occupied").innerText = stats.occupied || 0;
    if (document.getElementById("error")) document.getElementById("error").innerText = stats.error || 0;
    if (document.getElementById("supportCount")) document.getElementById("supportCount").innerText = stats.supportCount || 0;

    // Ép kiểu số chuẩn từ dữ liệu log thực tế từ Console
    const countFree = Number(stats.free || 0);
    const countOccupied = Number(stats.occupied || 0);
    const countError = Number(stats.error || 0);

    // 2. CẤU HÌNH BIỂU ĐỒ TRÒN (DOUGHNUT CHART)
    const ctxDoughnut = document.getElementById("chart");
    if (ctxDoughnut) {
        if (chartInstance) chartInstance.destroy();
        chartInstance = new Chart(ctxDoughnut, {
            type: 'doughnut',
            data: {
                labels: ['Trống', 'Đang dùng', 'Lỗi'],
                datasets: [{
                    data: [countFree, countOccupied, countError],
                    backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'], // Xanh, Cam, Đỏ
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    // 3. CẤU HÌNH BIỂU ĐỒ CỘT (BAR CHART)
    const ctxBar = document.getElementById("barChart");
    if (ctxBar) {
        if (barChartInstance) barChartInstance.destroy(); // Xóa chart cũ trước khi vẽ lại tuần hoàn
        barChartInstance = new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: ['Trống', 'Đang dùng', 'Lỗi'],
                datasets: [{
                    label: 'Số lượng tủ đồ',
                    data: [countFree, countOccupied, countError],
                    backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'], // Đồng bộ màu sắc 
                    borderRadius: 6, // Bo góc đầu cột cho hiện đại
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false } // Ẩn nhãn chú thích chung do màu cột đã rõ nghĩa
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1 } // Ép chia vạch theo số nguyên (1, 2, 3...)
                    }
                }
            }
        });
    }
    const lockers = await apiCall(ENDPOINTS.LOCKERS);
    if (document.getElementById("layout")) document.getElementById("layout").innerHTML = lockers.map(l => `<div class="locker status-${getStatusClass(l.status)}" title="Vị trí: ${l.location || 'N/A'}">${l.id}</div>`).join("");
    initTrendChartAndLogic();
}

async function loadTickets() {
    originalTicketsData = await apiCall(ENDPOINTS.TICKETS) || [];
    renderTicketDataToTable(originalTicketsData);
}

function renderTicketDataToTable(dataList) {
    const tbody = document.getElementById("supportTable");
    if (!tbody) return;

    // 🔥 THÊM LOGIC SẮP XẾP TĂNG DẦN (Ascending)
    if (dataList && dataList.length > 0) {
        dataList.sort((a, b) => {
            const timeA = new Date(a.created_at || a.createdAt || 0);
            const timeB = new Date(b.created_at || b.createdAt || 0);
            return timeA - timeB; // Sắp xếp tăng dần (Cũ nhất lên đầu)
        });
    }

    tbody.innerHTML = dataList?.map(t => `
        <tr>
            <td><strong>${t.id || 'N/A'}</strong></td>
            <td><span style="background: #e0f2fe; color: #0369a1; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 13px;">${t.locker?.id || 'N/A'}</span></td>
            <td style="font-family: monospace; font-weight: bold; color: #1e293b;">${t.session?.id || '[null]'}</td> 
            <td style="color: #475569; font-weight: 500;">${t.locker?.location || 'Chưa xác định'}</td>
            <td>${t.created_at || t.createdAt ? new Date(t.created_at || t.createdAt).toLocaleString('vi-VN') : '---'}</td>
            <td style="color:#ef4444; font-weight:500;">${t.reason || '---'}</td>
        </tr>
    `).join("") || `<tr><td colspan="6" style="text-align:center; padding:20px; color:#94a3b8;">Không tìm thấy dữ liệu.</td></tr>`;
}

function toggleTicketDateRange(value) {
    const rangeDiv = document.getElementById('ticketCustomDateRange');
    if (rangeDiv) {
        rangeDiv.style.display = (value === 'custom') ? 'flex' : 'none';
    }
}

// BẢN CẬP NHẬT: Thay thế hàm filterTicketTable cũ của Toàn
async function filterTicketTable() {
    // Ô 1: Lấy từ khóa (Mã phiên) từ id="ticketSearchInput"
    const keyword = document.getElementById("ticketSearchInput").value.trim();

    // Ô 2: Lấy vị trí từ id="ticketLocationInput"
    const location = document.getElementById("ticketLocationInput").value.trim();

    const timeRange = document.getElementById("ticketTimeRangeFilter")?.value || "ALL";

    let startStr = '';
    let endStr = '';
    const today = new Date();

    if (timeRange === 'today') {
        startStr = today.toISOString().split('T')[0];
        endStr = startStr;
    } else if (timeRange === 'month') {
        const firstDay = new Date(today.getFullYear(), today.getMonth(), 1);
        startStr = new Date(firstDay.getTime() - firstDay.getTimezoneOffset() * 60000).toISOString().split('T')[0];
        endStr = today.toISOString().split('T')[0];
    } else if (timeRange === 'year') {
        startStr = `${today.getFullYear()}-01-01`;
        endStr = today.toISOString().split('T')[0];
    } else if (timeRange === 'custom') {
        startStr = document.getElementById('ticketStartDate').value;
        endStr = document.getElementById('ticketEndDate').value;
        if (!startStr || !endStr) return alert("Vui lòng chọn đầy đủ ngày bắt đầu và kết thúc!");
        if (new Date(startStr) > new Date(endStr)) return alert("Ngày bắt đầu không thể lớn hơn ngày kết thúc!");
    }

    isSearching = (keyword !== "" || location !== "" || timeRange !== "ALL");

    // 🔥 URL này vẫn truyền đi 2 tham số: keyword (Backend nhận làm Mã phiên) và location (Vị trí)
    let url = `${ENDPOINTS.TICKETS}/search?keyword=${encodeURIComponent(keyword)}&location=${encodeURIComponent(location)}`;
    if (startStr && endStr) {
        url += `&startDate=${startStr}&endDate=${endStr}`;
    }

    console.log("ĐANG GỌI API LỌC TICKET:", url);
    const data = await apiCall(url);
    renderTicketDataToTable(data);
}

// BẢN CẬP NHẬT: Thay thế hàm clearTicketSearch cũ của Toàn
function clearTicketSearch() {
    document.getElementById("ticketSearchInput").value = "";
    if (document.getElementById("ticketTimeRangeFilter")) {
        document.getElementById("ticketTimeRangeFilter").value = "ALL";
    }

    const rangeDiv = document.getElementById('ticketCustomDateRange');
    if (rangeDiv) rangeDiv.style.display = 'none';

    document.getElementById('ticketStartDate').value = "";
    document.getElementById('ticketEndDate').value = "";

    isSearching = false;
    loadTickets(); // Gọi lại hàm load mặc định ban đầu để lấy lại toàn bộ danh sách từ DB
}

// ==========================================
// 4. QUẢN LÝ PHIÊN SỬ DỤNG (users.html)
// ==========================================
function toggleSessionDateRange(value) {
    const rangeDiv = document.getElementById('sessionCustomDateRange');
    if (rangeDiv) rangeDiv.style.display = (value === 'custom') ? 'flex' : 'none';
}

async function filterSessions() {
    const lid = document.getElementById("sessionSearch").value.trim();
    const status = document.getElementById("statusFilter").value;
    const timeRange = document.getElementById("timeRangeFilter")?.value || "ALL";
    const sortBy = document.getElementById("sortField").value;

    let startStr = '';
    let endStr = '';
    const today = new Date();

    if (timeRange === 'today') {
        startStr = today.toISOString().split('T')[0];
        endStr = startStr;
    } else if (timeRange === 'month') {
        const firstDay = new Date(today.getFullYear(), today.getMonth(), 1);
        startStr = new Date(firstDay.getTime() - firstDay.getTimezoneOffset() * 60000).toISOString().split('T')[0];
        endStr = today.toISOString().split('T')[0];
    } else if (timeRange === 'year') {
        startStr = `${today.getFullYear()}-01-01`;
        endStr = today.toISOString().split('T')[0];
    } else if (timeRange === 'custom') {
        startStr = document.getElementById('sessionStartDate').value;
        endStr = document.getElementById('sessionEndDate').value;
        if (!startStr || !endStr) return alert("Vui lòng chọn đầy đủ ngày bắt đầu và kết thúc!");
        if (new Date(startStr) > new Date(endStr)) return alert("Ngày bắt đầu không thể lớn hơn ngày kết thúc!");
    }

    isSearching = (lid !== "" || status !== "ALL" || timeRange !== "ALL");

    let url = `${ENDPOINTS.SESSIONS}/search?lockerId=${encodeURIComponent(lid)}&status=${status}&sortBy=${sortBy}`;
    if (startStr && endStr) url += `&startDate=${startStr}&endDate=${endStr}`;

    renderSessionTable(await apiCall(url));
}
async function loadSessions() { renderSessionTable(await apiCall(ENDPOINTS.SESSIONS)); }

function renderSessionTable(data) {
    const tbody = document.getElementById("sessionTable");
    if (!tbody) return;
    const list = Array.isArray(data) ? data : (data?.id ? [data] : []);

    // ĐỔI SỐ CỘT THÀNH 7 (Vì đã thêm cột vị trí)
    if (list.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px; color:#94a3b8;">Không tìm thấy dữ liệu phù hợp.</td></tr>`;
        return;
    }

    tbody.innerHTML = list.map(s => {
        const rawTime = s.start_time || s.startTime;
        const folderId = formatSessionFolderId(rawTime);

        // Nhận diện trường vị trí tủ từ Backend (Thường là lockerLocation)
        const locationStr = s.lockerLocation || s.location || 'Chưa xác định';

        return `
            <tr>
                <td><strong>${s.id}</strong></td>
                <td><span style="background: #e0f2fe; color: #0369a1; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 13px;">${s.lockerId || 'N/A'}</span></td>
                
                <td style="color: #475569; font-weight: 500;">${locationStr}</td>
                
                <td>${rawTime ? new Date(rawTime).toLocaleString('vi-VN') : '---'}</td>
                <td>${(s.end_time || s.endTime) ? new Date(s.end_time || s.endTime).toLocaleString('vi-VN') : '---'}</td>
                <td style="text-align: center; padding: 5px;">
                    ${folderId ? `<div id="img-container-${folderId}" style="width: 60px; height: 60px; border-radius: 6px; overflow: hidden; background: #e2e8f0; display: inline-flex; align-items: center; justify-content: center; border: 1px solid #cbd5e1;"><span style="font-size: 11px; color: #64748b;">⏳</span></div>` : '---'}
                </td>
                <td><span style="background: ${s.status === 'active' ? '#22c55e' : '#64748b'}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">${s.status}</span></td>
            </tr>
        `;
    }).join("");

    list.forEach(s => { if (s.start_time || s.startTime) fetchAndRenderRowImage(formatSessionFolderId(s.start_time || s.startTime)); });
}
async function fetchAndRenderRowImage(sid) {
    const container = document.getElementById(`img-container-${sid}`);
    if (!container) return;
    const imgs = await apiCall(`${API}/sessions/images-by-session?sessionId=${sid}`);
    if (imgs?.length > 0) {
        container.innerHTML = `<div style="position: relative; width: 100%; height: 100%;">
            <img src="${API}${imgs[0]}" style="width: 100%; height: 100%; object-fit: cover; cursor: pointer;" onclick="openImageGalleryPopup('${sid}', '${JSON.stringify(imgs).replace(/"/g, '&quot;')}')">
            ${imgs.length > 1 ? `<span style="position: absolute; bottom: 2px; right: 2px; background: rgba(0,0,0,0.7); color: white; font-size: 9px; padding: 1px 4px; border-radius: 4px; font-weight: bold;">+${imgs.length - 1}</span>` : ''}
        </div>`;
    } else container.innerHTML = `<span style="font-size: 16px;">🖐️</span>`;
}

function openImageGalleryPopup(sid, json) {
    const urls = JSON.parse(json.replace(/&quot;/g, '"'));
    let overlay = document.createElement("div"); overlay.id = "gallery-overlay"; overlay.style = "position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 9999; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 20px;";
    let content = document.createElement("div"); content.style = "background: white; padding: 20px; border-radius: 12px; max-width: 600px; width: 90%; box-shadow: 0 5px 25px rgba(0,0,0,0.3);";
    content.innerHTML = `<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;"><h3>Ảnh quét</h3><button onclick="document.getElementById('gallery-overlay').remove()" style="background: #ef4444; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-weight: bold;">Đóng ×</button></div><hr style="border: 0; border-top: 1px solid #e2e8f0; margin-bottom: 15px;"><div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 12px; max-height: 400px; overflow-y: auto; padding: 5px;">${urls.map(u => `<img src="${API}${u}" style="width: 100%; height: 130px; object-fit: cover; border-radius: 8px; border: 1px solid #cbd5e1; cursor: pointer;" onclick="window.open(this.src, '_blank')">`).join("")}</div>`;
    overlay.appendChild(content); overlay.onclick = (e) => { if (e.target.id === "gallery-overlay") overlay.remove(); };
    document.body.appendChild(overlay);
}

function resetSessionFilters() { document.getElementById("sessionSearch").value = ""; document.getElementById("statusFilter").value = "ALL"; if (document.getElementById("sortField")) document.getElementById("sortField").value = "start_time"; loadSessions(); }

// ==========================================
// LOGIC VẼ BIỂU ĐỒ XU HƯỚNG (TREND CHART)
// ==========================================


function initTrendChartAndLogic() {
    const trendCtx = document.getElementById('trendChart');
    if (!trendCtx) return; // Nếu đang ở trang khác thì bỏ qua

    if (!trendChartInstance) {
        // 1. Khởi tạo biểu đồ trống với 2 đường
        trendChartInstance = new Chart(trendCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Lượt sử dụng', data: [], borderColor: '#f39c12', backgroundColor: 'transparent', tension: 0.3, borderWidth: 2 },
                    { label: 'Ticket xử lý', data: [], borderColor: '#9b59b6', backgroundColor: 'transparent', tension: 0.3, borderWidth: 2 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { position: 'bottom' } },
                // BỔ SUNG ĐOẠN SCALES NÀY VÀO
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1, // Ép trục tung nhảy từng bước là 1 (chỉ hiện số nguyên)
                            precision: 0 // Không lấy số thập phân
                        }
                    }
                }
            }
        });

        // 2. Xử lý sự kiện khi đổi Dropdown
        document.getElementById('timeFilter')?.addEventListener('change', function () {

            if (this.value === 'custom') {
                document.getElementById('customDateRange').style.display = 'flex';
            } else {
                document.getElementById('customDateRange').style.display = 'none';
                fetchTrendData(this.value);
            }
        });

        // 3. Xử lý sự kiện khi bấm nút "Lọc" (cho tùy chọn ngày)
        document.getElementById('applyFilterBtn')?.addEventListener('click', function () {
            const start = document.getElementById('startDate').value;
            const end = document.getElementById('endDate').value;

            if (!start || !end) return alert("Vui lòng chọn đầy đủ ngày bắt đầu và kết thúc!");
            if (new Date(start) > new Date(end)) return alert("Ngày bắt đầu không thể lớn hơn ngày kết thúc!");

            fetchTrendData('custom', start, end);
        });
    }

    // 4. Gọi API lấy dữ liệu mặc định (7 ngày qua) khi vừa vào trang
    fetchTrendData('week');
}

// Hàm gọi API lấy dữ liệu Trend và vẽ lại biểu đồ
async function fetchTrendData(type, start = '', end = '') {
    // Đảm bảo biến API đã được định nghĩa ở đầu file (VD: const API = "http://localhost:8080/api";)
    let url = `${API}/api/dashboard/trend?type=${type}`;
    if (type === 'custom') url += `&startDate=${start}&endDate=${end}`;

    const data = await apiCall(url);
    if (data && trendChartInstance) {
        // Cập nhật dữ liệu mới vào biểu đồ
        trendChartInstance.data.labels = data.labels;
        trendChartInstance.data.datasets[0].data = data.usage;
        trendChartInstance.data.datasets[1].data = data.tickets;

        // Vẽ lại biểu đồ
        trendChartInstance.update();

    }
}