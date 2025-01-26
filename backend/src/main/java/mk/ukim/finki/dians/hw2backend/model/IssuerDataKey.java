package mk.ukim.finki.dians.hw2backend.model;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.time.LocalDate;
import java.util.Objects;

@AllArgsConstructor
@Data
@NoArgsConstructor
public class IssuerDataKey implements Serializable {
    private LocalDate date;
    private String issuer;

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IssuerDataKey that = (IssuerDataKey) o;
        return Objects.equals(date, that.date) && Objects.equals(issuer, that.issuer);
    }

    @Override
    public int hashCode() {
        return Objects.hash(date, issuer);
    }
}
